TEMPLATES = {
    'chatbot': (
        "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
                "{{ message['content'] }}"
            "{% elif message['role'] == 'user' %}"
                "{{ '\n\nHuman: ' + message['content'] +  eos_token }}"
            "{% elif message['role'] == 'assistant' %}"
                "{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}"
            "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
            "{{ '\n\nAssistant: ' }}"
        "{% endif %}"
    ),
    'qa': (
        "{% if instruction %}"
            "{{ 'Instruction: ' + instruction + '\n\n' }}"
        "{% endif %}"
        "{% if question %}"
            "{{ 'Question: ' + question + '\n\n' }}"
        "{% endif %}"
        "{% if answer %}"
            "{{ 'Answer: ' + answer + eos_token }}"
        "{% elif add_generation_prompt %}"
            "{{ 'Answer: ' }}"
        "{% endif %}"
    ),
    'summarization': (
        "{% if article %}"
            "{{ 'Article: ' + article + '\n\n' }}"
        "{% endif %}"
        "{% if summary %}"
            "{{ 'Summary: ' + summary + eos_token }}"
        "{% elif add_generation_prompt %}"
            "{{ 'Summary: ' }}"
        "{% endif %}"
    ),
    'instruction': (
        "{% if instruction %}"
            "{{ 'Instruction: ' + instruction + '\n\n' }}"
        "{% endif %}"
        "{% if response %}"
            "{{ 'Response: ' + response + eos_token }}"
        "{% elif add_generation_prompt %}"
            "{{ 'Response: ' }}"
        "{% endif %}"
    )
}