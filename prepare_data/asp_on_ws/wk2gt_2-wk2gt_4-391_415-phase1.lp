
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

	orientation(wk2gt_2,141,391). coords(wk2gt_2,98,63,391).  visible(wk2gt_2,391). happensAt(inactive(wk2gt_2),391). orientation(wk2gt_4,149,391). coords(wk2gt_4,144,117,391).  visible(wk2gt_4,391). happensAt(walking(wk2gt_4),391).
orientation(wk2gt_2,141,392). coords(wk2gt_2,98,63,392).  visible(wk2gt_2,392). happensAt(inactive(wk2gt_2),392). orientation(wk2gt_4,149,392). coords(wk2gt_4,142,117,392).  visible(wk2gt_4,392). happensAt(walking(wk2gt_4),392).
orientation(wk2gt_2,141,393). coords(wk2gt_2,98,63,393).  visible(wk2gt_2,393). happensAt(inactive(wk2gt_2),393). orientation(wk2gt_4,149,393). coords(wk2gt_4,139,116,393).  visible(wk2gt_4,393). happensAt(walking(wk2gt_4),393).
orientation(wk2gt_2,141,394). coords(wk2gt_2,98,63,394).  visible(wk2gt_2,394). happensAt(inactive(wk2gt_2),394). orientation(wk2gt_4,149,394). coords(wk2gt_4,138,116,394).  visible(wk2gt_4,394). happensAt(walking(wk2gt_4),394).
orientation(wk2gt_2,141,395). coords(wk2gt_2,98,63,395).  visible(wk2gt_2,395). happensAt(inactive(wk2gt_2),395). orientation(wk2gt_4,149,395). coords(wk2gt_4,137,115,395).  visible(wk2gt_4,395). happensAt(walking(wk2gt_4),395).
orientation(wk2gt_2,141,396). coords(wk2gt_2,98,63,396).  visible(wk2gt_2,396). happensAt(inactive(wk2gt_2),396). orientation(wk2gt_4,149,396). coords(wk2gt_4,136,115,396).  visible(wk2gt_4,396). happensAt(walking(wk2gt_4),396).
orientation(wk2gt_2,141,397). coords(wk2gt_2,98,63,397).  visible(wk2gt_2,397). happensAt(inactive(wk2gt_2),397). orientation(wk2gt_4,149,397). coords(wk2gt_4,136,115,397).  visible(wk2gt_4,397). happensAt(walking(wk2gt_4),397).
orientation(wk2gt_2,141,398). coords(wk2gt_2,98,63,398).  visible(wk2gt_2,398). happensAt(inactive(wk2gt_2),398). orientation(wk2gt_4,149,398). coords(wk2gt_4,136,115,398).  visible(wk2gt_4,398). happensAt(walking(wk2gt_4),398).
orientation(wk2gt_2,141,399). coords(wk2gt_2,98,63,399).  visible(wk2gt_2,399). happensAt(inactive(wk2gt_2),399). orientation(wk2gt_4,149,399). coords(wk2gt_4,135,114,399).  visible(wk2gt_4,399). happensAt(walking(wk2gt_4),399).
orientation(wk2gt_2,141,400). coords(wk2gt_2,98,63,400).  visible(wk2gt_2,400). happensAt(inactive(wk2gt_2),400). orientation(wk2gt_4,149,400). coords(wk2gt_4,133,114,400).  visible(wk2gt_4,400). happensAt(walking(wk2gt_4),400).
orientation(wk2gt_2,141,401). coords(wk2gt_2,98,63,401).  visible(wk2gt_2,401). happensAt(inactive(wk2gt_2),401). orientation(wk2gt_4,149,401). coords(wk2gt_4,133,114,401).  visible(wk2gt_4,401). happensAt(walking(wk2gt_4),401).
orientation(wk2gt_2,141,402). coords(wk2gt_2,98,63,402).  visible(wk2gt_2,402). happensAt(inactive(wk2gt_2),402). orientation(wk2gt_4,149,402). coords(wk2gt_4,130,112,402).  visible(wk2gt_4,402). happensAt(walking(wk2gt_4),402).
orientation(wk2gt_2,141,403). coords(wk2gt_2,98,63,403).  visible(wk2gt_2,403). happensAt(inactive(wk2gt_2),403). orientation(wk2gt_4,146,403). coords(wk2gt_4,128,112,403).  visible(wk2gt_4,403). happensAt(walking(wk2gt_4),403).
orientation(wk2gt_2,141,404). coords(wk2gt_2,98,63,404).  visible(wk2gt_2,404). happensAt(inactive(wk2gt_2),404). orientation(wk2gt_4,146,404). coords(wk2gt_4,127,110,404).  visible(wk2gt_4,404). happensAt(walking(wk2gt_4),404).
orientation(wk2gt_2,141,405). coords(wk2gt_2,98,63,405).  visible(wk2gt_2,405). happensAt(inactive(wk2gt_2),405). orientation(wk2gt_4,146,405). coords(wk2gt_4,126,110,405).  visible(wk2gt_4,405). happensAt(walking(wk2gt_4),405).
orientation(wk2gt_2,141,406). coords(wk2gt_2,98,63,406).  visible(wk2gt_2,406). happensAt(inactive(wk2gt_2),406). orientation(wk2gt_4,146,406). coords(wk2gt_4,125,109,406).  visible(wk2gt_4,406). happensAt(walking(wk2gt_4),406).
orientation(wk2gt_2,141,407). coords(wk2gt_2,98,63,407).  visible(wk2gt_2,407). happensAt(inactive(wk2gt_2),407). orientation(wk2gt_4,149,407). coords(wk2gt_4,125,108,407).  visible(wk2gt_4,407). happensAt(walking(wk2gt_4),407).
orientation(wk2gt_2,141,408). coords(wk2gt_2,98,63,408).  visible(wk2gt_2,408). happensAt(inactive(wk2gt_2),408). orientation(wk2gt_4,149,408). coords(wk2gt_4,124,108,408).  visible(wk2gt_4,408). happensAt(walking(wk2gt_4),408).
orientation(wk2gt_2,141,409). coords(wk2gt_2,98,63,409).  visible(wk2gt_2,409). happensAt(inactive(wk2gt_2),409). orientation(wk2gt_4,149,409). coords(wk2gt_4,124,108,409).  visible(wk2gt_4,409). happensAt(walking(wk2gt_4),409).
orientation(wk2gt_2,141,410). coords(wk2gt_2,98,63,410).  visible(wk2gt_2,410). happensAt(inactive(wk2gt_2),410). orientation(wk2gt_4,149,410). coords(wk2gt_4,124,108,410).  visible(wk2gt_4,410). happensAt(walking(wk2gt_4),410).
orientation(wk2gt_2,141,411). coords(wk2gt_2,98,63,411).  visible(wk2gt_2,411). happensAt(inactive(wk2gt_2),411). orientation(wk2gt_4,149,411). coords(wk2gt_4,123,107,411).  visible(wk2gt_4,411). happensAt(walking(wk2gt_4),411).
orientation(wk2gt_2,141,412). coords(wk2gt_2,98,63,412).  visible(wk2gt_2,412). happensAt(inactive(wk2gt_2),412). orientation(wk2gt_4,149,412). coords(wk2gt_4,122,107,412).  visible(wk2gt_4,412). happensAt(walking(wk2gt_4),412).
orientation(wk2gt_2,141,413). coords(wk2gt_2,98,63,413).  visible(wk2gt_2,413). happensAt(inactive(wk2gt_2),413). orientation(wk2gt_4,149,413). coords(wk2gt_4,121,107,413).  visible(wk2gt_4,413). happensAt(walking(wk2gt_4),413).
orientation(wk2gt_2,141,414). coords(wk2gt_2,98,63,414).  visible(wk2gt_2,414). happensAt(inactive(wk2gt_2),414). orientation(wk2gt_4,149,414). coords(wk2gt_4,120,107,414).  visible(wk2gt_4,414). happensAt(walking(wk2gt_4),414).
orientation(wk2gt_2,141,415). coords(wk2gt_2,98,63,415).  visible(wk2gt_2,415). happensAt(inactive(wk2gt_2),415). orientation(wk2gt_4,149,415). coords(wk2gt_4,118,106,415).  visible(wk2gt_4,415). happensAt(walking(wk2gt_4),415).
#show holdsAt/2.