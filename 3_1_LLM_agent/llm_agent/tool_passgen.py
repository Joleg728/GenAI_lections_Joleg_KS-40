# llm_agent/tool_passgen.py

import secrets as scrts
import string

class PassGen:
    name = "pass_gen"
    description = "Создаёт по запросу надёжный пароль с настройками."

    def use(self, length : int = 16, sp_symb : bool = True, numbs : bool = True):
        """
        Создаёт надёжный пароль.
        """ 
        try:

            symb_bank = string.ascii_letters

            if numbs:
                symb_bank += string.digits
            
            if sp_symb:
                symb_bank += string.punctuation
            
            psswrd = ''.join(scrts.choice(symb_bank) for i in range(length))
            
            return psswrd

        except Exception as e:
            # Это сообщение будет выведено в лог, если ошибка возникнет на самом верхнем уровне
            print(f"> Ошибка")
            return f"Произошла ошибка': {e}"