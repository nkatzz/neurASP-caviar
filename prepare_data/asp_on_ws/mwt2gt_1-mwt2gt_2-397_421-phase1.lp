
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

	orientation(mwt2gt_1,146,397). coords(mwt2gt_1,109,89,397).  visible(mwt2gt_1,397). happensAt(walking(mwt2gt_1),397). orientation(mwt2gt_2,144,397). coords(mwt2gt_2,104,113,397).  visible(mwt2gt_2,397). happensAt(walking(mwt2gt_2),397).
orientation(mwt2gt_1,146,398). coords(mwt2gt_1,108,90,398).  visible(mwt2gt_1,398). happensAt(walking(mwt2gt_1),398). orientation(mwt2gt_2,144,398). coords(mwt2gt_2,104,113,398).  visible(mwt2gt_2,398). happensAt(walking(mwt2gt_2),398).
orientation(mwt2gt_1,146,399). coords(mwt2gt_1,108,90,399).  visible(mwt2gt_1,399). happensAt(walking(mwt2gt_1),399). orientation(mwt2gt_2,144,399). coords(mwt2gt_2,103,113,399).  visible(mwt2gt_2,399). happensAt(walking(mwt2gt_2),399).
orientation(mwt2gt_1,146,400). coords(mwt2gt_1,107,90,400).  visible(mwt2gt_1,400). happensAt(walking(mwt2gt_1),400). orientation(mwt2gt_2,144,400). coords(mwt2gt_2,102,113,400).  visible(mwt2gt_2,400). happensAt(walking(mwt2gt_2),400).
orientation(mwt2gt_1,146,401). coords(mwt2gt_1,105,90,401).  visible(mwt2gt_1,401). happensAt(walking(mwt2gt_1),401). orientation(mwt2gt_2,144,401). coords(mwt2gt_2,102,113,401).  visible(mwt2gt_2,401). happensAt(walking(mwt2gt_2),401).
orientation(mwt2gt_1,150,402). coords(mwt2gt_1,105,90,402).  visible(mwt2gt_1,402). happensAt(walking(mwt2gt_1),402). orientation(mwt2gt_2,144,402). coords(mwt2gt_2,102,113,402).  visible(mwt2gt_2,402). happensAt(walking(mwt2gt_2),402).
orientation(mwt2gt_1,150,403). coords(mwt2gt_1,103,90,403).  visible(mwt2gt_1,403). happensAt(walking(mwt2gt_1),403). orientation(mwt2gt_2,149,403). coords(mwt2gt_2,102,114,403).  visible(mwt2gt_2,403). happensAt(walking(mwt2gt_2),403).
orientation(mwt2gt_1,150,404). coords(mwt2gt_1,102,90,404).  visible(mwt2gt_1,404). happensAt(walking(mwt2gt_1),404). orientation(mwt2gt_2,149,404). coords(mwt2gt_2,101,114,404).  visible(mwt2gt_2,404). happensAt(walking(mwt2gt_2),404).
orientation(mwt2gt_1,150,405). coords(mwt2gt_1,102,90,405).  visible(mwt2gt_1,405). happensAt(walking(mwt2gt_1),405). orientation(mwt2gt_2,149,405). coords(mwt2gt_2,100,114,405).  visible(mwt2gt_2,405). happensAt(walking(mwt2gt_2),405).
orientation(mwt2gt_1,150,406). coords(mwt2gt_1,101,90,406).  visible(mwt2gt_1,406). happensAt(walking(mwt2gt_1),406). orientation(mwt2gt_2,149,406). coords(mwt2gt_2,100,114,406).  visible(mwt2gt_2,406). happensAt(walking(mwt2gt_2),406).
orientation(mwt2gt_1,150,407). coords(mwt2gt_1,101,90,407).  visible(mwt2gt_1,407). happensAt(walking(mwt2gt_1),407). orientation(mwt2gt_2,149,407). coords(mwt2gt_2,98,114,407).  visible(mwt2gt_2,407). happensAt(walking(mwt2gt_2),407).
orientation(mwt2gt_1,150,408). coords(mwt2gt_1,101,90,408).  visible(mwt2gt_1,408). happensAt(walking(mwt2gt_1),408). orientation(mwt2gt_2,149,408). coords(mwt2gt_2,97,115,408).  visible(mwt2gt_2,408). happensAt(walking(mwt2gt_2),408).
orientation(mwt2gt_1,150,409). coords(mwt2gt_1,101,90,409).  visible(mwt2gt_1,409). happensAt(walking(mwt2gt_1),409). orientation(mwt2gt_2,149,409). coords(mwt2gt_2,96,115,409).  visible(mwt2gt_2,409). happensAt(walking(mwt2gt_2),409).
orientation(mwt2gt_1,150,410). coords(mwt2gt_1,100,90,410).  visible(mwt2gt_1,410). happensAt(walking(mwt2gt_1),410). orientation(mwt2gt_2,149,410). coords(mwt2gt_2,95,116,410).  visible(mwt2gt_2,410). happensAt(walking(mwt2gt_2),410).
orientation(mwt2gt_1,150,411). coords(mwt2gt_1,99,90,411).  visible(mwt2gt_1,411). happensAt(walking(mwt2gt_1),411). orientation(mwt2gt_2,149,411). coords(mwt2gt_2,93,116,411).  visible(mwt2gt_2,411). happensAt(walking(mwt2gt_2),411).
orientation(mwt2gt_1,150,412). coords(mwt2gt_1,99,90,412).  visible(mwt2gt_1,412). happensAt(walking(mwt2gt_1),412). orientation(mwt2gt_2,149,412). coords(mwt2gt_2,93,118,412).  visible(mwt2gt_2,412). happensAt(walking(mwt2gt_2),412).
orientation(mwt2gt_1,150,413). coords(mwt2gt_1,98,90,413).  visible(mwt2gt_1,413). happensAt(walking(mwt2gt_1),413). orientation(mwt2gt_2,149,413). coords(mwt2gt_2,93,119,413).  visible(mwt2gt_2,413). happensAt(walking(mwt2gt_2),413).
orientation(mwt2gt_1,150,414). coords(mwt2gt_1,96,90,414).  visible(mwt2gt_1,414). happensAt(walking(mwt2gt_1),414). orientation(mwt2gt_2,149,414). coords(mwt2gt_2,93,119,414).  visible(mwt2gt_2,414). happensAt(walking(mwt2gt_2),414).
orientation(mwt2gt_1,150,415). coords(mwt2gt_1,96,90,415).  visible(mwt2gt_1,415). happensAt(walking(mwt2gt_1),415). orientation(mwt2gt_2,149,415). coords(mwt2gt_2,93,119,415).  visible(mwt2gt_2,415). happensAt(walking(mwt2gt_2),415).
orientation(mwt2gt_1,150,416). coords(mwt2gt_1,95,90,416).  visible(mwt2gt_1,416). happensAt(walking(mwt2gt_1),416). orientation(mwt2gt_2,149,416). coords(mwt2gt_2,93,119,416).  visible(mwt2gt_2,416). happensAt(walking(mwt2gt_2),416).
orientation(mwt2gt_1,150,417). coords(mwt2gt_1,95,90,417).  visible(mwt2gt_1,417). happensAt(walking(mwt2gt_1),417). orientation(mwt2gt_2,149,417). coords(mwt2gt_2,92,119,417).  visible(mwt2gt_2,417). happensAt(walking(mwt2gt_2),417).
orientation(mwt2gt_1,150,418). coords(mwt2gt_1,95,90,418).  visible(mwt2gt_1,418). happensAt(walking(mwt2gt_1),418). orientation(mwt2gt_2,149,418). coords(mwt2gt_2,91,120,418).  visible(mwt2gt_2,418). happensAt(walking(mwt2gt_2),418).
orientation(mwt2gt_1,150,419). coords(mwt2gt_1,94,90,419).  visible(mwt2gt_1,419). happensAt(walking(mwt2gt_1),419). orientation(mwt2gt_2,149,419). coords(mwt2gt_2,90,120,419).  visible(mwt2gt_2,419). happensAt(walking(mwt2gt_2),419).
orientation(mwt2gt_1,150,420). coords(mwt2gt_1,94,89,420).  visible(mwt2gt_1,420). happensAt(walking(mwt2gt_1),420). orientation(mwt2gt_2,149,420). coords(mwt2gt_2,90,122,420).  visible(mwt2gt_2,420). happensAt(walking(mwt2gt_2),420).
orientation(mwt2gt_1,150,421). coords(mwt2gt_1,94,89,421).  visible(mwt2gt_1,421). happensAt(walking(mwt2gt_1),421). orientation(mwt2gt_2,149,421). coords(mwt2gt_2,90,122,421).  visible(mwt2gt_2,421). happensAt(walking(mwt2gt_2),421).
#show holdsAt/2.