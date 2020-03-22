"""
This module contains the Connector class,
where every data fetching should begin with instantiating
the Connector class.
"""
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from jinja2 import Environment, StrictUndefined, Template
from requests import Request, Response, Session

from ..errors import UnreachableError
from .config_manager import config_directory, ensure_config
from .errors import RequestError
from .implicit_database import ImplicitDatabase, ImplicitTable


INFO_TEMPLATE = Template(
    """{% for tb in tbs.keys() %}
Table {{dbname}}.{{tb}}

Parameters
----------
{% if tbs[tb].required_params %}{{", ".join(tbs[tb].required_params)}} required {% endif %}
{% if tbs[tb].optional_params %}{{", ".join(tbs[tb].optional_params)}} optional {% endif %}

Examples
--------
>>> dc.query({{", ".join(["\\\"{}\\\"".format(tb)] + tbs[tb].joined_query_fields)}})
>>> dc.show_schema("{{tb}}")
{% endfor %}
"""
)


class Connector:
    """
    The main class of DataConnector.
    """

    impdb: ImplicitDatabase
    vars: Dict[str, Any]
    auth_params: Dict[str, Any]
    session: Session
    jenv: Environment

    def __init__(
        self,
        config_path: str,
        auth_params: Optional[Dict[str, Any]] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Connector

        parameters
        ----------
        config_path : str
            The path to the config. It can be hosted, e.g. "yelp", or from
            local filesystem, e.g. "./yelp"
        **kwargs : Dict[str, Any]
            Additional parameters
        """

        self.session = Session()
        if (
            config_path.startswith(".")
            or config_path.startswith("/")
            or config_path.startswith("~")
        ):
            path = Path(config_path).resolve()
            self.impdb = ImplicitDatabase(path)
        else:
            # From Github!
            ensure_config(config_path)
            path = config_directory() / config_path
            self.impdb = ImplicitDatabase(path)
        # print(self.impdb.tables)

        self.vars = kwargs
        self.auth_params = auth_params or {}
        self.jenv = Environment(undefined=StrictUndefined)

    def _fetch(
        self,
        table: ImplicitTable,
        auth_params: Optional[Dict[str, Any]],
        kwargs: Dict[str, Any],
    ) -> Response:
        method = table.method
        url = table.url
        req_data: Dict[str, Dict[str, Any]] = {
            "headers": {},
            "params": {},
            "cookies": {},
        }

        merged_vars = {**self.vars, **kwargs}

        count_key = self.impdb.config["request"]["paganition"]["count_key"]
        if "returned_number" in merged_vars:
            merged_vars[count_key] = merged_vars.pop("returned_number")

        if table.authorization is not None:
            table.authorization.build(req_data, auth_params or self.auth_params)
        for key in ["headers", "params", "cookies"]:
            if getattr(table, key) is not None:
                instantiated_fields = getattr(table, key).populate(
                    self.jenv, merged_vars
                )
                req_data[key].update(**instantiated_fields)
        if table.body is not None:
            # TODO: do we support binary body?
            instantiated_fields = table.body.populate(self.jenv, merged_vars)
            if table.body_ctype == "application/x-www-form-urlencoded":
                req_data["data"] = instantiated_fields
            elif table.body_ctype == "application/json":
                req_data["json"] = instantiated_fields
            else:
                raise UnreachableError

        resp: Response = self.session.send(  # type: ignore
            Request(
                method=method,
                url=url,
                headers=req_data["headers"],
                params=req_data["params"],
                json=req_data.get("json"),
                data=req_data.get("data"),
                cookies=req_data["cookies"],
            ).prepare()
        )

        if resp.status_code != 200:
            raise RequestError(status_code=resp.status_code, message=resp.text)

        return resp

    def query(
        self,
        table: str,
        auth_params: Optional[Dict[str, Any]] = None,
        **where: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Query the API to get a table.

        Parameters
        ----------
        table : str
            The table name.
        auth_params : Optional[Dict[str, Any]] = None
            The parameters for authentication. Usually the authentication parameters
            should be defined when instantiating the Connector. In case some tables have different
            authentication options, a different authentication parameter can be defined here.
            This parameter will override the one from Connector if passed.
        **where: Dict[str, Any]
            The additional parameters required for the query.

        Returns
        -------
            pd.DataFrame
        """
        assert table in self.impdb.tables, f"No such table {table} in {self.impdb.name}"

        itable = self.impdb.tables[table]

        max_count = self.impdb.config["request"]["paganition"]["max_count"]
        df = []

        pag_type = self.impdb.config["request"]["paganition"]["type"]

        if "returned_number" not in where:
            where["returned_number"] = max_count
            resp = self._fetch(itable, auth_params, where)
            df = itable.from_response(resp)

        else:
            n_page = math.ceil(where["returned_number"] / max_count)
            remain = where["returned_number"] % max_count
            if pag_type == "cursor":
                last_id = 0
            for i in range(n_page):
                if i < n_page - 1:
                    where["returned_number"] = max_count
                else:
                    if remain != 0:
                        where["returned_number"] = remain
                    else:
                        where["returned_number"] = max_count
                if pag_type == "cursor" and i > 0:
                    where[self.impdb.config["request"]["paganition"]["cursor_key"]] = (
                        last_id - 1
                    )
                if pag_type == "limit":
                    where[self.impdb.config["request"]["paganition"]["anchor_key"]] = (
                        0 + i * max_count
                    )

                resp = self._fetch(itable, auth_params, where)
                df_ = itable.from_response(resp)
                if pag_type == "cursor":
                    last_id = (
                        int(
                            df_[
                                self.impdb.config["request"]["paganition"]["cursor_id"]
                            ][len(df_) - 1]
                        )
                        - 1
                    )
                if i == 0:
                    df = df_.copy()
                else:
                    df = pd.concat([df, df_], axis=0)
            df.reset_index(drop=True, inplace=True)

        return df

    @property
    def table_names(self) -> List[str]:
        """
        Return all the table names contained in this database.
        """
        return list(self.impdb.tables.keys())

    @property
    def info(self) -> None:
        """
        Show the information of a website and guide users how to issue queries
        """

        # get info
        tbs: Dict[str, Any] = {}
        for cur_table in self.impdb.tables.keys():
            table_config_content = self.impdb.tables[cur_table].config
            params_required = []
            params_optional = []
            example_query_fields = []
            count = 1
            for k, val in table_config_content["request"]["params"].items():
                if isinstance(val, bool) and val:
                    params_required.append(k)
                    example_query_fields.append(f"""{k}="word{count}\"""")
                    count += 1
                elif isinstance(val, bool):
                    params_optional.append(k)
            tbs[cur_table] = {}
            tbs[cur_table]["required_params"] = params_required
            tbs[cur_table]["optional_params"] = params_optional
            tbs[cur_table]["joined_query_fields"] = example_query_fields

        # show table info
        print(
            INFO_TEMPLATE.render(
                ntables=len(self.table_names), dbname=self.impdb.name, tbs=tbs
            )
        )

    def show_schema(self, table_name: str) -> pd.DataFrame:
        """
        Show the returned schema of a table

        Parameters
        ----------
        table_name : str
            The table name.

        Returns
        -------
            pd.DataFrame
        """
        print(f"table: {table_name}")
        table_config_content = self.impdb.tables[table_name].config
        schema = table_config_content["response"]["schema"]
        new_schema_dict: Dict[str, List[Any]] = {}
        new_schema_dict["column_name"] = []
        new_schema_dict["data_type"] = []
        for k in schema.keys():
            new_schema_dict["column_name"].append(k)
            new_schema_dict["data_type"].append(schema[k]["type"])
        return pd.DataFrame.from_dict(new_schema_dict)
