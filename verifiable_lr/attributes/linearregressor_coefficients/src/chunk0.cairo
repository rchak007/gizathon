use orion::numbers::{FixedTrait, FP16x16};

fn compute(ref a: Array<FP16x16>) {
a.append(FP16x16 { mag: 51272, sign: true });
a.append(FP16x16 { mag: 936407, sign: true });
a.append(FP16x16 { mag: 3520520, sign: true });
a.append(FP16x16 { mag: 1640186, sign: true });
a.append(FP16x16 { mag: 13774273, sign: true });
a.append(FP16x16 { mag: 13302149, sign: false });
a.append(FP16x16 { mag: 2494210, sign: false });
a.append(FP16x16 { mag: 9390783, sign: false });
a.append(FP16x16 { mag: 29275366, sign: false });
a.append(FP16x16 { mag: 430069, sign: false });
}