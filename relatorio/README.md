# Relatórios

| Arquivo | Conteúdo |
|---|---|
| `relatorio_N2.pdf` | Relatório final da N2: é o arquivo enviado no Ambiente Virtual |
| `relatorio_N2.docx` | Fonte editável do relatório da N2 |
| `relatorio_N1.pdf` | Relatório da primeira entrega (N1) |
| `template_fci.docx` | Template oficial da FCI convertido para `.docx`; é a base do relatório |
| `roteiro_video.md` | Roteiro do vídeo de até 5 minutos e o passo a passo de publicação |

O `.docx` e o PDF são gerados por `src/report.py` a partir de
`results/metrics/*.json`: os números do texto não são digitados à mão, e
`tests/test_resultados.py` confere que eles batem com os resultados.

```bash
python -m src.report --pdf                                       # sem o link do vídeo
python -m src.report --video-url https://youtu.be/SEU_LINK --pdf # versão final
```

A exportação para PDF usa o Microsoft Word por automação, então só funciona no
Windows com Word instalado. Sem Word, abra o `.docx` e exporte como PDF.
