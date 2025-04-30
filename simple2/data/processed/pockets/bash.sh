cd data/processed/pockets

for pdbfile in *.pdb
do
    base_name="${pdbfile%.pdb}"        # 去掉.pdb后缀
    pqrfile="${base_name}.pqr"
    if [ ! -f "$pqrfile" ]; then
        echo "⚡ 生成$pqrfile"
        pdb2pqr --ff=AMBER --with-ph=7.0  "$pdbfile" "$pqrfile"
    else
        echo "✅ 已存在$pqrfile，跳过"
    fi
done

echo "🎯 PQR补齐完成！"
