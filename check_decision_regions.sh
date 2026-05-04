#!/bin/bash

SCRIPT="cosine_decision_regions.py"
SEEDS=5
N_VALUES="3 5 10"
D_MAX=6

if [ ! -f "$SCRIPT" ]; then
    echo "Error: $SCRIPT not found"
    exit 1
fi

pass=0
fail=0
total=0

for n in $N_VALUES; do
    d_limit=$((n < D_MAX ? n : D_MAX))
    for d in $(seq 2 $d_limit); do
        for cosine in "" "--use-cosine"; do
            for braid in "" "--use-braid"; do
				cosine_label=${cosine:-"             "}
                braid_label=${braid:-"           "}
				# If our matrix is full rank, multiplying by the braid matrix
				# discards information, so we should ignore this case
				if [ -n "$braid" ] && [ $d -eq $n ]; then
					continue
				fi
                for seed in $(seq 1 $SEEDS); do
                    ((total++))
                    if python "$SCRIPT" --n $n --d $d $cosine $braid --seed $seed 2>/dev/null; then
                        ((pass++))
                    else
						echo "FAIL: n=$n d=$d $cosine_label $braid_label seed=$seed"
                        ((fail++))
                    fi
					printf "\r[%d passed, %d failed] n=%d d=%d %s %s seed=%d" $pass $fail $n $d "$cosine_label" "$braid_label" $seed
                done
            done
        done
    done
done

echo ""
echo "Results: $pass passed, $fail failed out of $total"
