import sqlite3

class SqliteDB:
    def __init__(self, db_path: str):
        """
        初始化数据库连接
        :param db_path: SQLite 数据库文件路径，如 'test.db'
        """
        self.connection = sqlite3.connect(db_path)
        self.connection.row_factory = sqlite3.Row  # 支持按列名访问数据
        self.cursor = self.connection.cursor()

    def execute(self, sql: str, params: tuple = None):
        """
        执行 SQL 语句
        :param sql: SQL语句
        :param params: 参数化查询参数（防止SQL注入）
        :return: 游标对象
        """
        if params:
            self.cursor.execute(sql, params)
        else:
            self.cursor.execute(sql)
        return self.cursor

    def executemany(self, sql: str, params_list: list):
        """
        批量执行 SQL 语句
        :param sql: SQL语句
        :param params_list: 参数列表
        """
        self.cursor.executemany(sql, params_list)

    def fetchone(self, sql: str, params: tuple = None):
        """
        查询单条记录
        :param sql: 查询语句
        :param params: 参数
        :return: 单条记录（字典形式）
        """
        cursor = self.execute(sql, params)
        result = cursor.fetchone()
        cursor.close()
        return dict(result) if result else None

    def fetchall(self, sql: str, params: tuple = None):
        """
        查询所有记录
        :param sql: 查询语句
        :param params: 参数
        :return: 列表，每项为一条记录的字典
        """
        cursor = self.execute(sql, params)
        results = cursor.fetchall()
        cursor.close()
        return [dict(row) for row in results]

    def insert(self, table: str, data: dict):
        """
        插入数据
        :param table: 表名
        :param data: 要插入的数据字典 {字段: 值}
        :return: 最后插入的行ID
        """
        columns = ', '.join(data.keys())
        placeholders = ', '.join('?' * len(data))
        values = tuple(data.values())
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        self.execute(sql, values)
        self.connection.commit()
        return self.cursor.lastrowid

    def update(self, table: str, data: dict, where: str, params: tuple = None):
        """
        更新数据
        :param table: 表名
        :param data: 要更新的数据字典 {字段: 值}
        :param where: WHERE 条件语句字符串，例如 "id = ?"
        :param params: WHERE 的参数（可选）
        """
        set_clause = ', '.join([f"{key} = ?" for key in data.keys()])
        values = tuple(data.values())
        if params:
            values += params
        sql = f"UPDATE {table} SET {set_clause} WHERE {where}"
        self.execute(sql, values)
        self.connection.commit()

    def delete(self, table: str, where: str, params: tuple = None):
        """
        删除数据
        :param table: 表名
        :param where: WHERE 条件语句
        :param params: 参数（可选）
        """
        sql = f"DELETE FROM {table} WHERE {where}"
        self.execute(sql, params)
        self.connection.commit()

    def create_table(self, sql: str):
        """
        创建表
        :param sql: CREATE TABLE 语句
        """
        self.execute(sql)
        self.connection.commit()

    def commit(self):
        """提交事务"""
        self.connection.commit()

    def close(self):
        """关闭连接"""
        self.cursor.close()
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()