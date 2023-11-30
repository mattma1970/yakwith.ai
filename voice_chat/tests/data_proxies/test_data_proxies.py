from voice_chat.utils import DataProxy

def test_get_all():
    assert (
        len(DataProxy.get_all("data_proxy_tests", "./")) == 2
    ), "Get_all returned wrong number of entries."

def test_client_authorized():
    assert (
        DataProxy.verify_client_authorized("user_1_key", "data_proxy_tests", "./")
        == True
    )
    "Failed client_authorized"

def test_client_not_authorized():
    assert (
        DataProxy.verify_client_authorized("user_1_key_2", "data_proxy_tests", "./")
        == False
    ), "Returned true for user that was not authorized."

def test_service_configured():
    assert (
        DataProxy.get_3p_service_configs(
            "user_1_key", "data_proxy_tests", "assemblyai", "data_proxy_tests", "./"
        )["api_token"]
        == "assembly_token"
    ), "Failed service_config_test"
