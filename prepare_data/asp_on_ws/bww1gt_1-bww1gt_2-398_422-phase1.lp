
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

	orientation(bww1gt_1,179,398). coords(bww1gt_1,105,264,398).  visible(bww1gt_1,398). happensAt(inactive(bww1gt_1),398). orientation(bww1gt_2,155,398). coords(bww1gt_2,110,146,398).  visible(bww1gt_2,398). happensAt(walking(bww1gt_2),398).
orientation(bww1gt_1,1,399). coords(bww1gt_1,105,264,399).  visible(bww1gt_1,399). happensAt(inactive(bww1gt_1),399). orientation(bww1gt_2,154,399). coords(bww1gt_2,112,146,399).  visible(bww1gt_2,399). happensAt(walking(bww1gt_2),399).
orientation(bww1gt_1,4,400). coords(bww1gt_1,105,264,400).  visible(bww1gt_1,400). happensAt(inactive(bww1gt_1),400). orientation(bww1gt_2,158,400). coords(bww1gt_2,114,146,400).  visible(bww1gt_2,400). happensAt(walking(bww1gt_2),400).
orientation(bww1gt_1,1,401). coords(bww1gt_1,105,264,401).  visible(bww1gt_1,401). happensAt(inactive(bww1gt_1),401). orientation(bww1gt_2,158,401). coords(bww1gt_2,115,146,401).  visible(bww1gt_2,401). happensAt(walking(bww1gt_2),401).
orientation(bww1gt_1,4,402). coords(bww1gt_1,105,264,402).  visible(bww1gt_1,402). happensAt(inactive(bww1gt_1),402). orientation(bww1gt_2,157,402). coords(bww1gt_2,115,146,402).  visible(bww1gt_2,402). happensAt(walking(bww1gt_2),402).
orientation(bww1gt_1,5,403). coords(bww1gt_1,105,264,403).  visible(bww1gt_1,403). happensAt(inactive(bww1gt_1),403). orientation(bww1gt_2,159,403). coords(bww1gt_2,116,146,403).  visible(bww1gt_2,403). happensAt(walking(bww1gt_2),403).
orientation(bww1gt_1,0,404). coords(bww1gt_1,105,264,404).  visible(bww1gt_1,404). happensAt(inactive(bww1gt_1),404). orientation(bww1gt_2,158,404). coords(bww1gt_2,117,146,404).  visible(bww1gt_2,404). happensAt(walking(bww1gt_2),404).
orientation(bww1gt_1,3,405). coords(bww1gt_1,105,264,405).  visible(bww1gt_1,405). happensAt(inactive(bww1gt_1),405). orientation(bww1gt_2,158,405). coords(bww1gt_2,118,146,405).  visible(bww1gt_2,405). happensAt(walking(bww1gt_2),405).
orientation(bww1gt_1,4,406). coords(bww1gt_1,105,264,406).  visible(bww1gt_1,406). happensAt(inactive(bww1gt_1),406). orientation(bww1gt_2,158,406). coords(bww1gt_2,118,146,406).  visible(bww1gt_2,406). happensAt(walking(bww1gt_2),406).
orientation(bww1gt_1,8,407). coords(bww1gt_1,105,264,407).  visible(bww1gt_1,407). happensAt(inactive(bww1gt_1),407). orientation(bww1gt_2,158,407). coords(bww1gt_2,119,146,407).  visible(bww1gt_2,407). happensAt(walking(bww1gt_2),407).
orientation(bww1gt_1,179,408). coords(bww1gt_1,105,264,408).  visible(bww1gt_1,408). happensAt(inactive(bww1gt_1),408). orientation(bww1gt_2,157,408). coords(bww1gt_2,119,146,408).  visible(bww1gt_2,408). happensAt(walking(bww1gt_2),408).
orientation(bww1gt_1,2,409). coords(bww1gt_1,105,264,409).  visible(bww1gt_1,409). happensAt(inactive(bww1gt_1),409). orientation(bww1gt_2,156,409). coords(bww1gt_2,119,146,409).  visible(bww1gt_2,409). happensAt(walking(bww1gt_2),409).
orientation(bww1gt_1,2,410). coords(bww1gt_1,105,264,410).  visible(bww1gt_1,410). happensAt(inactive(bww1gt_1),410). orientation(bww1gt_2,160,410). coords(bww1gt_2,120,146,410).  visible(bww1gt_2,410). happensAt(walking(bww1gt_2),410).
orientation(bww1gt_1,3,411). coords(bww1gt_1,105,264,411).  visible(bww1gt_1,411). happensAt(inactive(bww1gt_1),411). orientation(bww1gt_2,155,411). coords(bww1gt_2,120,146,411).  visible(bww1gt_2,411). happensAt(walking(bww1gt_2),411).
orientation(bww1gt_1,5,412). coords(bww1gt_1,105,264,412).  visible(bww1gt_1,412). happensAt(inactive(bww1gt_1),412). orientation(bww1gt_2,158,412). coords(bww1gt_2,120,146,412).  visible(bww1gt_2,412). happensAt(walking(bww1gt_2),412).
orientation(bww1gt_1,6,413). coords(bww1gt_1,105,264,413).  visible(bww1gt_1,413). happensAt(inactive(bww1gt_1),413). orientation(bww1gt_2,155,413). coords(bww1gt_2,121,146,413).  visible(bww1gt_2,413). happensAt(walking(bww1gt_2),413).
orientation(bww1gt_1,3,414). coords(bww1gt_1,105,264,414).  visible(bww1gt_1,414). happensAt(inactive(bww1gt_1),414). orientation(bww1gt_2,159,414). coords(bww1gt_2,123,146,414).  visible(bww1gt_2,414). happensAt(walking(bww1gt_2),414).
orientation(bww1gt_1,1,415). coords(bww1gt_1,105,264,415).  visible(bww1gt_1,415). happensAt(inactive(bww1gt_1),415). orientation(bww1gt_2,161,415). coords(bww1gt_2,125,146,415).  visible(bww1gt_2,415). happensAt(walking(bww1gt_2),415).
orientation(bww1gt_1,7,416). coords(bww1gt_1,105,264,416).  visible(bww1gt_1,416). happensAt(inactive(bww1gt_1),416). orientation(bww1gt_2,157,416). coords(bww1gt_2,127,146,416).  visible(bww1gt_2,416). happensAt(walking(bww1gt_2),416).
orientation(bww1gt_1,1,417). coords(bww1gt_1,105,264,417).  visible(bww1gt_1,417). happensAt(inactive(bww1gt_1),417). orientation(bww1gt_2,156,417). coords(bww1gt_2,129,146,417).  visible(bww1gt_2,417). happensAt(walking(bww1gt_2),417).
orientation(bww1gt_1,4,418). coords(bww1gt_1,105,264,418).  visible(bww1gt_1,418). happensAt(inactive(bww1gt_1),418). orientation(bww1gt_2,159,418). coords(bww1gt_2,130,146,418).  visible(bww1gt_2,418). happensAt(walking(bww1gt_2),418).
orientation(bww1gt_1,4,419). coords(bww1gt_1,105,264,419).  visible(bww1gt_1,419). happensAt(inactive(bww1gt_1),419). orientation(bww1gt_2,158,419). coords(bww1gt_2,131,146,419).  visible(bww1gt_2,419). happensAt(walking(bww1gt_2),419).
orientation(bww1gt_1,3,420). coords(bww1gt_1,105,264,420).  visible(bww1gt_1,420). happensAt(inactive(bww1gt_1),420). orientation(bww1gt_2,157,420). coords(bww1gt_2,132,146,420).  visible(bww1gt_2,420). happensAt(walking(bww1gt_2),420).
orientation(bww1gt_1,179,421). coords(bww1gt_1,105,264,421).  visible(bww1gt_1,421). happensAt(inactive(bww1gt_1),421). orientation(bww1gt_2,155,421). coords(bww1gt_2,132,146,421).  visible(bww1gt_2,421). happensAt(walking(bww1gt_2),421).
orientation(bww1gt_1,2,422). coords(bww1gt_1,105,264,422).  visible(bww1gt_1,422). happensAt(inactive(bww1gt_1),422). orientation(bww1gt_2,154,422). coords(bww1gt_2,133,145,422).  visible(bww1gt_2,422). happensAt(walking(bww1gt_2),422).
#show holdsAt/2.