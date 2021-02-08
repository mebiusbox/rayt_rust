#![allow(dead_code)]

mod rayt;
mod code100;
mod code101;
mod code102;
mod code103;
mod code104;
mod code105;
mod code106;
mod code107;
mod code108;
mod code109;
mod code110;
mod code111;
mod code112;
mod code113;
mod code114;
mod code115;
mod code116;
mod code117;
mod code201;
mod code202;
mod code203;
mod code204;
mod code205;
mod code206;
mod code207;
mod code208;
mod code209;
mod code301;
mod code302;
mod code303;
mod code304;
mod code305;
mod code306;
mod code307;
mod code308;
mod code309;
mod code310;
mod code311;

fn main() {
    let mut no = 0;
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        no = args[1].parse::<i32>().unwrap();
    }
    run(no);
}

fn run(no: i32) {
    match no {
        0 => code311::run(), // final code no: 311
        100 => code100::run(),
        101 => code101::run(),
        102 => code102::run(),
        103 => code103::run(),
        104 => code104::run(),
        105 => code105::run(),
        106 => code106::run(),
        107 => code107::run(),
        108 => code108::run(),
        109 => code109::run(),
        110 => code110::run(),
        111 => code111::run(),
        112 => code112::run(),
        113 => code113::run(),
        114 => code114::run(),
        115 => code115::run(),
        116 => code116::run(),
        117 => code117::run(),
        201 => code201::run(),
        202 => code202::run(),
        203 => code203::run(),
        204 => code204::run(),
        205 => code205::run(),
        206 => code206::run(),
        207 => code207::run(),
        208 => code208::run(),
        209 => code209::run(),
        301 => code301::run(),
        302 => code302::run(),
        303 => code303::run(),
        304 => code304::run(),
        305 => code305::run(),
        306 => code306::run(),
        307 => code307::run(),
        308 => code308::run(),
        309 => code309::run(),
        310 => code310::run(),
        311 => code311::run(),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // cargo test all -- --nocapture --ignored
    // If to check the rendered image, comment out line 5 in window.rs
    #[test] #[ignore]
    fn all() {
        for i in 100..=311 {
            println!("code{} running...", i);
            run(i);

            // let output_path = std::path::Path::new("render.png");
            // if output_path.exists() {
            //     let rename_path = std::path::Path::new("render").join(format!("render{}.png", i));
            //     std::fs::rename("render.png", rename_path).unwrap();
            // }
        }
    }

    #[test] #[ignore]
    fn main() {
        run(0);
    }
}
