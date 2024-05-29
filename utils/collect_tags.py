# Define the file names
input_file = '/home/ldy/Misc/TransformerNER/ner-data/train_TAG.txt'
output_file = 'tag_statistics.txt'

# Initialize a dictionary to store the count of each tag
tag_counts = {}

# Read the input file and count the tags
with open(input_file, 'r') as file:
    for line in file:
        tags = line.strip().split()
        for tag in tags:
            if tag in tag_counts:
                tag_counts[tag] += 1
            else:
                tag_counts[tag] = 1

# Print the statistics
print("Tag Statistics:")
for tag, count in tag_counts.items():
    print(f"{tag}: {count}")

# Save the statistics to the output file
with open(output_file, 'w') as file:
    file.write("Tag Statistics:\n")
    for tag, count in tag_counts.items():
        file.write(f"{tag}: {count}\n")
