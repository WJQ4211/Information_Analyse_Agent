"""
情报分析智能体 - 数据连接器基类
Intelligence Analysis Agent - Connector Base Class

所有 OSINT 数据源连接器的抽象基类。
"""

from abc import ABC, abstractmethod

from ingestion.models import Document


class BaseConnector(ABC):
    """
    数据连接器抽象基类

    所有 OSINT 数据源（网页、PDF、RSS 等）都必须实现此接口。
    """

    @abstractmethod
    async def fetch(self, source: str) -> Document:
        """
        从指定来源采集文档

        Args:
            source: 来源标识（URL、文件路径等）

        Returns:
            Document 对象

        Raises:
            ValueError: 来源无效
            ConnectionError: 网络请求失败
        """
        ...

    @abstractmethod
    def validate_source(self, source: str) -> bool:
        """
        验证来源标识是否合法

        Args:
            source: 来源标识

        Returns:
            True 表示合法，False 表示不合法
        """
        ...
