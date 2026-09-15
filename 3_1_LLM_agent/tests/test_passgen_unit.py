# tests/test_passgen_unit.py
from llm_agent.tool_passgen import PassGen

def test_passgen_length_and_charset():
    """Проверяет, что пароль имеет нужную длину и содержит только разрешённые символы."""
    pwd = PassGen().use(length=13, sp_symb=True, numbs=False)
    assert len(pwd) == 13
    assert any(c.isalpha() for c in pwd)          # есть буквы
    assert any(c in string.punctuation for c in pwd)  # есть спецсимволы
    assert not any(c.isdigit() for c in pwd)      # нет цифр

def test_passgen_no_special_chars():
    """Проверяет генерацию без спецсимволов."""
    pwd = PassGen().use(length=20, sp_symb=False, numbs=True)
    assert len(pwd) == 20
    assert pwd.isalnum()                          # только буквы и цифры
    assert any(c.isdigit() for c in pwd)

def test_passgen_length_validation():
    """Проверяет, что при length < 1 возвращается ошибка."""
    result = PassGen().use(length=0)
    assert "Ошибка" in result
    assert "Длина пароля должна быть не менее 1 символа" in result