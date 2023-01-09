
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

	orientation(ms3ggt_0,152,392). coords(ms3ggt_0,142,138,392).  visible(ms3ggt_0,392). happensAt(active(ms3ggt_0),392). orientation(ms3ggt_2,168,392). coords(ms3ggt_2,151,200,392).  visible(ms3ggt_2,392). happensAt(walking(ms3ggt_2),392).
orientation(ms3ggt_0,152,393). coords(ms3ggt_0,142,138,393).  visible(ms3ggt_0,393). happensAt(active(ms3ggt_0),393). orientation(ms3ggt_2,168,393). coords(ms3ggt_2,150,200,393).  visible(ms3ggt_2,393). happensAt(walking(ms3ggt_2),393).
orientation(ms3ggt_0,152,394). coords(ms3ggt_0,141,138,394).  visible(ms3ggt_0,394). happensAt(active(ms3ggt_0),394). orientation(ms3ggt_2,168,394). coords(ms3ggt_2,149,199,394).  visible(ms3ggt_2,394). happensAt(walking(ms3ggt_2),394).
orientation(ms3ggt_0,152,395). coords(ms3ggt_0,141,138,395).  visible(ms3ggt_0,395). happensAt(walking(ms3ggt_0),395). orientation(ms3ggt_2,168,395). coords(ms3ggt_2,148,199,395).  visible(ms3ggt_2,395). happensAt(walking(ms3ggt_2),395).
orientation(ms3ggt_0,158,396). coords(ms3ggt_0,141,138,396).  visible(ms3ggt_0,396). happensAt(walking(ms3ggt_0),396). orientation(ms3ggt_2,168,396). coords(ms3ggt_2,146,199,396).  visible(ms3ggt_2,396). happensAt(walking(ms3ggt_2),396).
orientation(ms3ggt_0,158,397). coords(ms3ggt_0,140,138,397).  visible(ms3ggt_0,397). happensAt(walking(ms3ggt_0),397). orientation(ms3ggt_2,168,397). coords(ms3ggt_2,145,199,397).  visible(ms3ggt_2,397). happensAt(walking(ms3ggt_2),397).
orientation(ms3ggt_0,158,398). coords(ms3ggt_0,140,138,398).  visible(ms3ggt_0,398). happensAt(walking(ms3ggt_0),398). orientation(ms3ggt_2,168,398). coords(ms3ggt_2,142,196,398).  visible(ms3ggt_2,398). happensAt(walking(ms3ggt_2),398).
orientation(ms3ggt_0,158,399). coords(ms3ggt_0,139,138,399).  visible(ms3ggt_0,399). happensAt(walking(ms3ggt_0),399). orientation(ms3ggt_2,168,399). coords(ms3ggt_2,141,196,399).  visible(ms3ggt_2,399). happensAt(walking(ms3ggt_2),399).
orientation(ms3ggt_0,158,400). coords(ms3ggt_0,139,138,400).  visible(ms3ggt_0,400). happensAt(walking(ms3ggt_0),400). orientation(ms3ggt_2,168,400). coords(ms3ggt_2,140,195,400).  visible(ms3ggt_2,400). happensAt(walking(ms3ggt_2),400).
orientation(ms3ggt_0,161,401). coords(ms3ggt_0,139,138,401).  visible(ms3ggt_0,401). happensAt(walking(ms3ggt_0),401). orientation(ms3ggt_2,168,401). coords(ms3ggt_2,139,194,401).  visible(ms3ggt_2,401). happensAt(walking(ms3ggt_2),401).
orientation(ms3ggt_0,161,402). coords(ms3ggt_0,138,138,402).  visible(ms3ggt_0,402). happensAt(walking(ms3ggt_0),402). orientation(ms3ggt_2,168,402). coords(ms3ggt_2,138,194,402).  visible(ms3ggt_2,402). happensAt(walking(ms3ggt_2),402).
orientation(ms3ggt_0,161,403). coords(ms3ggt_0,137,138,403).  visible(ms3ggt_0,403). happensAt(walking(ms3ggt_0),403). orientation(ms3ggt_2,168,403). coords(ms3ggt_2,137,193,403).  visible(ms3ggt_2,403). happensAt(walking(ms3ggt_2),403).
orientation(ms3ggt_0,156,404). coords(ms3ggt_0,136,139,404).  visible(ms3ggt_0,404). happensAt(walking(ms3ggt_0),404). orientation(ms3ggt_2,159,404). coords(ms3ggt_2,137,193,404).  visible(ms3ggt_2,404). happensAt(walking(ms3ggt_2),404).
orientation(ms3ggt_0,156,405). coords(ms3ggt_0,134,139,405).  visible(ms3ggt_0,405). happensAt(walking(ms3ggt_0),405). orientation(ms3ggt_2,159,405). coords(ms3ggt_2,137,191,405).  visible(ms3ggt_2,405). happensAt(walking(ms3ggt_2),405).
orientation(ms3ggt_0,156,406). coords(ms3ggt_0,133,139,406).  visible(ms3ggt_0,406). happensAt(walking(ms3ggt_0),406). orientation(ms3ggt_2,159,406). coords(ms3ggt_2,136,190,406).  visible(ms3ggt_2,406). happensAt(walking(ms3ggt_2),406).
orientation(ms3ggt_0,156,407). coords(ms3ggt_0,132,140,407).  visible(ms3ggt_0,407). happensAt(walking(ms3ggt_0),407). orientation(ms3ggt_2,159,407). coords(ms3ggt_2,136,190,407).  visible(ms3ggt_2,407). happensAt(walking(ms3ggt_2),407).
orientation(ms3ggt_0,158,408). coords(ms3ggt_0,131,140,408).  visible(ms3ggt_0,408). happensAt(walking(ms3ggt_0),408). orientation(ms3ggt_2,159,408). coords(ms3ggt_2,135,189,408).  visible(ms3ggt_2,408). happensAt(walking(ms3ggt_2),408).
orientation(ms3ggt_0,158,409). coords(ms3ggt_0,131,140,409).  visible(ms3ggt_0,409). happensAt(walking(ms3ggt_0),409). orientation(ms3ggt_2,159,409). coords(ms3ggt_2,132,187,409).  visible(ms3ggt_2,409). happensAt(walking(ms3ggt_2),409).
orientation(ms3ggt_0,158,410). coords(ms3ggt_0,131,140,410).  visible(ms3ggt_0,410). happensAt(walking(ms3ggt_0),410). orientation(ms3ggt_2,159,410). coords(ms3ggt_2,130,186,410).  visible(ms3ggt_2,410). happensAt(walking(ms3ggt_2),410).
orientation(ms3ggt_0,158,411). coords(ms3ggt_0,130,141,411).  visible(ms3ggt_0,411). happensAt(walking(ms3ggt_0),411). orientation(ms3ggt_2,159,411). coords(ms3ggt_2,129,185,411).  visible(ms3ggt_2,411). happensAt(walking(ms3ggt_2),411).
orientation(ms3ggt_0,158,412). coords(ms3ggt_0,130,141,412).  visible(ms3ggt_0,412). happensAt(walking(ms3ggt_0),412). orientation(ms3ggt_2,159,412). coords(ms3ggt_2,127,183,412).  visible(ms3ggt_2,412). happensAt(walking(ms3ggt_2),412).
orientation(ms3ggt_0,158,413). coords(ms3ggt_0,130,141,413).  visible(ms3ggt_0,413). happensAt(walking(ms3ggt_0),413). orientation(ms3ggt_2,159,413). coords(ms3ggt_2,126,182,413).  visible(ms3ggt_2,413). happensAt(walking(ms3ggt_2),413).
orientation(ms3ggt_0,162,414). coords(ms3ggt_0,128,142,414).  visible(ms3ggt_0,414). happensAt(walking(ms3ggt_0),414). orientation(ms3ggt_2,162,414). coords(ms3ggt_2,126,181,414).  visible(ms3ggt_2,414). happensAt(walking(ms3ggt_2),414).
orientation(ms3ggt_0,162,415). coords(ms3ggt_0,126,142,415).  visible(ms3ggt_0,415). happensAt(walking(ms3ggt_0),415). orientation(ms3ggt_2,162,415). coords(ms3ggt_2,126,179,415).  visible(ms3ggt_2,415). happensAt(walking(ms3ggt_2),415).
orientation(ms3ggt_0,162,416). coords(ms3ggt_0,125,142,416).  visible(ms3ggt_0,416). happensAt(walking(ms3ggt_0),416). orientation(ms3ggt_2,162,416). coords(ms3ggt_2,125,178,416).  visible(ms3ggt_2,416). happensAt(walking(ms3ggt_2),416).
#show holdsAt/2.