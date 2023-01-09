
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(ms3ggt_1,162,392). coords(ms3ggt_1,144,158,392).  visible(ms3ggt_1,392). happensAt(walking(ms3ggt_1),392). orientation(ms3ggt_2,168,392). coords(ms3ggt_2,151,200,392).  visible(ms3ggt_2,392). happensAt(walking(ms3ggt_2),392).
orientation(ms3ggt_1,153,393). coords(ms3ggt_1,144,158,393).  visible(ms3ggt_1,393). happensAt(walking(ms3ggt_1),393). orientation(ms3ggt_2,168,393). coords(ms3ggt_2,150,200,393).  visible(ms3ggt_2,393). happensAt(walking(ms3ggt_2),393).
orientation(ms3ggt_1,153,394). coords(ms3ggt_1,147,158,394).  visible(ms3ggt_1,394). happensAt(walking(ms3ggt_1),394). orientation(ms3ggt_2,168,394). coords(ms3ggt_2,149,199,394).  visible(ms3ggt_2,394). happensAt(walking(ms3ggt_2),394).
orientation(ms3ggt_1,153,395). coords(ms3ggt_1,147,158,395).  visible(ms3ggt_1,395). happensAt(walking(ms3ggt_1),395). orientation(ms3ggt_2,168,395). coords(ms3ggt_2,148,199,395).  visible(ms3ggt_2,395). happensAt(walking(ms3ggt_2),395).
orientation(ms3ggt_1,153,396). coords(ms3ggt_1,149,158,396).  visible(ms3ggt_1,396). happensAt(walking(ms3ggt_1),396). orientation(ms3ggt_2,168,396). coords(ms3ggt_2,146,199,396).  visible(ms3ggt_2,396). happensAt(walking(ms3ggt_2),396).
orientation(ms3ggt_1,153,397). coords(ms3ggt_1,149,158,397).  visible(ms3ggt_1,397). happensAt(walking(ms3ggt_1),397). orientation(ms3ggt_2,168,397). coords(ms3ggt_2,145,199,397).  visible(ms3ggt_2,397). happensAt(walking(ms3ggt_2),397).
orientation(ms3ggt_1,153,398). coords(ms3ggt_1,149,158,398).  visible(ms3ggt_1,398). happensAt(walking(ms3ggt_1),398). orientation(ms3ggt_2,168,398). coords(ms3ggt_2,142,196,398).  visible(ms3ggt_2,398). happensAt(walking(ms3ggt_2),398).
orientation(ms3ggt_1,153,399). coords(ms3ggt_1,149,158,399).  visible(ms3ggt_1,399). happensAt(walking(ms3ggt_1),399). orientation(ms3ggt_2,168,399). coords(ms3ggt_2,141,196,399).  visible(ms3ggt_2,399). happensAt(walking(ms3ggt_2),399).
orientation(ms3ggt_1,153,400). coords(ms3ggt_1,150,158,400).  visible(ms3ggt_1,400). happensAt(walking(ms3ggt_1),400). orientation(ms3ggt_2,168,400). coords(ms3ggt_2,140,195,400).  visible(ms3ggt_2,400). happensAt(walking(ms3ggt_2),400).
orientation(ms3ggt_1,153,401). coords(ms3ggt_1,150,159,401).  visible(ms3ggt_1,401). happensAt(walking(ms3ggt_1),401). orientation(ms3ggt_2,168,401). coords(ms3ggt_2,139,194,401).  visible(ms3ggt_2,401). happensAt(walking(ms3ggt_2),401).
orientation(ms3ggt_1,153,402). coords(ms3ggt_1,150,159,402).  visible(ms3ggt_1,402). happensAt(walking(ms3ggt_1),402). orientation(ms3ggt_2,168,402). coords(ms3ggt_2,138,194,402).  visible(ms3ggt_2,402). happensAt(walking(ms3ggt_2),402).
orientation(ms3ggt_1,153,403). coords(ms3ggt_1,150,159,403).  visible(ms3ggt_1,403). happensAt(walking(ms3ggt_1),403). orientation(ms3ggt_2,168,403). coords(ms3ggt_2,137,193,403).  visible(ms3ggt_2,403). happensAt(walking(ms3ggt_2),403).
orientation(ms3ggt_1,153,404). coords(ms3ggt_1,151,159,404).  visible(ms3ggt_1,404). happensAt(walking(ms3ggt_1),404). orientation(ms3ggt_2,159,404). coords(ms3ggt_2,137,193,404).  visible(ms3ggt_2,404). happensAt(walking(ms3ggt_2),404).
orientation(ms3ggt_1,153,405). coords(ms3ggt_1,151,159,405).  visible(ms3ggt_1,405). happensAt(walking(ms3ggt_1),405). orientation(ms3ggt_2,159,405). coords(ms3ggt_2,137,191,405).  visible(ms3ggt_2,405). happensAt(walking(ms3ggt_2),405).
orientation(ms3ggt_1,153,406). coords(ms3ggt_1,152,159,406).  visible(ms3ggt_1,406). happensAt(walking(ms3ggt_1),406). orientation(ms3ggt_2,159,406). coords(ms3ggt_2,136,190,406).  visible(ms3ggt_2,406). happensAt(walking(ms3ggt_2),406).
orientation(ms3ggt_1,153,407). coords(ms3ggt_1,152,159,407).  visible(ms3ggt_1,407). happensAt(walking(ms3ggt_1),407). orientation(ms3ggt_2,159,407). coords(ms3ggt_2,136,190,407).  visible(ms3ggt_2,407). happensAt(walking(ms3ggt_2),407).
orientation(ms3ggt_1,153,408). coords(ms3ggt_1,152,159,408).  visible(ms3ggt_1,408). happensAt(walking(ms3ggt_1),408). orientation(ms3ggt_2,159,408). coords(ms3ggt_2,135,189,408).  visible(ms3ggt_2,408). happensAt(walking(ms3ggt_2),408).
orientation(ms3ggt_1,153,409). coords(ms3ggt_1,152,159,409).  visible(ms3ggt_1,409). happensAt(walking(ms3ggt_1),409). orientation(ms3ggt_2,159,409). coords(ms3ggt_2,132,187,409).  visible(ms3ggt_2,409). happensAt(walking(ms3ggt_2),409).
orientation(ms3ggt_1,153,410). coords(ms3ggt_1,153,158,410).  visible(ms3ggt_1,410). happensAt(walking(ms3ggt_1),410). orientation(ms3ggt_2,159,410). coords(ms3ggt_2,130,186,410).  visible(ms3ggt_2,410). happensAt(walking(ms3ggt_2),410).
orientation(ms3ggt_1,153,411). coords(ms3ggt_1,154,158,411).  visible(ms3ggt_1,411). happensAt(walking(ms3ggt_1),411). orientation(ms3ggt_2,159,411). coords(ms3ggt_2,129,185,411).  visible(ms3ggt_2,411). happensAt(walking(ms3ggt_2),411).
orientation(ms3ggt_1,153,412). coords(ms3ggt_1,158,159,412).  visible(ms3ggt_1,412). happensAt(walking(ms3ggt_1),412). orientation(ms3ggt_2,159,412). coords(ms3ggt_2,127,183,412).  visible(ms3ggt_2,412). happensAt(walking(ms3ggt_2),412).
orientation(ms3ggt_1,153,413). coords(ms3ggt_1,159,159,413).  visible(ms3ggt_1,413). happensAt(walking(ms3ggt_1),413). orientation(ms3ggt_2,159,413). coords(ms3ggt_2,126,182,413).  visible(ms3ggt_2,413). happensAt(walking(ms3ggt_2),413).
orientation(ms3ggt_1,160,414). coords(ms3ggt_1,161,158,414).  visible(ms3ggt_1,414). happensAt(walking(ms3ggt_1),414). orientation(ms3ggt_2,162,414). coords(ms3ggt_2,126,181,414).  visible(ms3ggt_2,414). happensAt(walking(ms3ggt_2),414).
orientation(ms3ggt_1,160,415). coords(ms3ggt_1,163,158,415).  visible(ms3ggt_1,415). happensAt(walking(ms3ggt_1),415). orientation(ms3ggt_2,162,415). coords(ms3ggt_2,126,179,415).  visible(ms3ggt_2,415). happensAt(walking(ms3ggt_2),415).
orientation(ms3ggt_1,160,416). coords(ms3ggt_1,165,158,416).  visible(ms3ggt_1,416). happensAt(walking(ms3ggt_1),416). orientation(ms3ggt_2,162,416). coords(ms3ggt_2,125,178,416).  visible(ms3ggt_2,416). happensAt(walking(ms3ggt_2),416).
#show holdsAt/2.