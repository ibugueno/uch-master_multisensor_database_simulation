python generate_lists.py --base_root /ruta/a --output_dir /ruta/a/output_lists

## Ejemplo

python generate_lists.py --base_root ../../input/ --output_dir tmp/

wc -l tmp/*

for f in tmp/*.txt; do first_line=$(head -n 1 "$f"); echo "$f:"; echo "$first_line"; [ -e "$first_line" ] && echo "✔️ Exists" || echo "❌ NOT FOUND"; echo; done