# llm_agent/tool_passgen.py

import secrets as scrts
import string

class PassGen:
    name = "pass_gen"
    description = "Создаёт по запросу надёжный пароль с настройками."

    def use(self, length : int = 16, sp_symb : bool = True, numbs : bool = True):

        if length < 1:
            return "Ошибка: Длина пароля должна быть не менее 1 символа."

        try:

            letters = string.ascii_letters
            digits = string.digits if numbs else ""
            punctuation = string.punctuation if sp_symb else ""

            symb_bank = letters + digits + punctuation
            
            while True:

                psswrd = ''.join(scrts.choice(symb_bank) for i in range(length))

                has_digits = not numbs or any(c in digits for c in psswrd)
                has_punctuation = not sp_symb or any(c in punctuation for c in psswrd)
                has_letters = any(c in letters for c in psswrd)

                # Если все условия выполнены, возвращаем пароль
                if has_digits and has_punctuation and has_letters:
                    return psswrd
            
                

        except Exception as e:
            # Это сообщение будет выведено в лог, если ошибка возникнет на самом верхнем уровне
            return f"Произошла ошибка: {e}"