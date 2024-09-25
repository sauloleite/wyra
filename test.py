from wyra.data_maker import FineTuningDataMaker

fine_tunner = FineTuningDataMaker()

content = """Software Engineering is the systematic application of engineering principles to the design, development, testing, and maintenance of software. 
It involves a structured process of producing reliable and efficient software that meets specified requirements within a given timeline and budget.
"""

formatted_content = fine_tunner.format_data(content)

print(formatted_content)