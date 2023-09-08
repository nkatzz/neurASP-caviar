
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

	orientation(lb1gt_1,151,396). coords(lb1gt_1,166,170,396).  visible(lb1gt_1,396). happensAt(active(lb1gt_1),396). orientation(lb1gt_2,144,396). coords(lb1gt_2,189,141,396).  visible(lb1gt_2,396). happensAt(active(lb1gt_2),396).
orientation(lb1gt_1,151,397). coords(lb1gt_1,166,169,397).  visible(lb1gt_1,397). happensAt(active(lb1gt_1),397). orientation(lb1gt_2,144,397). coords(lb1gt_2,189,141,397).  visible(lb1gt_2,397). happensAt(active(lb1gt_2),397).
orientation(lb1gt_1,151,398). coords(lb1gt_1,166,169,398).  visible(lb1gt_1,398). happensAt(active(lb1gt_1),398). orientation(lb1gt_2,144,398). coords(lb1gt_2,189,141,398).  visible(lb1gt_2,398). happensAt(active(lb1gt_2),398).
orientation(lb1gt_1,151,399). coords(lb1gt_1,166,168,399).  visible(lb1gt_1,399). happensAt(active(lb1gt_1),399). orientation(lb1gt_2,144,399). coords(lb1gt_2,189,141,399).  visible(lb1gt_2,399). happensAt(active(lb1gt_2),399).
orientation(lb1gt_1,151,400). coords(lb1gt_1,166,168,400).  visible(lb1gt_1,400). happensAt(active(lb1gt_1),400). orientation(lb1gt_2,144,400). coords(lb1gt_2,189,141,400).  visible(lb1gt_2,400). happensAt(active(lb1gt_2),400).
orientation(lb1gt_1,151,401). coords(lb1gt_1,166,168,401).  visible(lb1gt_1,401). happensAt(active(lb1gt_1),401). orientation(lb1gt_2,144,401). coords(lb1gt_2,189,141,401).  visible(lb1gt_2,401). happensAt(active(lb1gt_2),401).
orientation(lb1gt_1,151,402). coords(lb1gt_1,166,168,402).  visible(lb1gt_1,402). happensAt(active(lb1gt_1),402). orientation(lb1gt_2,144,402). coords(lb1gt_2,189,141,402).  visible(lb1gt_2,402). happensAt(active(lb1gt_2),402).
orientation(lb1gt_1,151,403). coords(lb1gt_1,166,168,403).  visible(lb1gt_1,403). happensAt(active(lb1gt_1),403). orientation(lb1gt_2,144,403). coords(lb1gt_2,189,141,403).  visible(lb1gt_2,403). happensAt(active(lb1gt_2),403).
orientation(lb1gt_1,151,404). coords(lb1gt_1,166,168,404).  visible(lb1gt_1,404). happensAt(active(lb1gt_1),404). orientation(lb1gt_2,144,404). coords(lb1gt_2,189,141,404).  visible(lb1gt_2,404). happensAt(active(lb1gt_2),404).
orientation(lb1gt_1,151,405). coords(lb1gt_1,166,168,405).  visible(lb1gt_1,405). happensAt(active(lb1gt_1),405). orientation(lb1gt_2,144,405). coords(lb1gt_2,189,141,405).  visible(lb1gt_2,405). happensAt(active(lb1gt_2),405).
orientation(lb1gt_1,151,406). coords(lb1gt_1,166,168,406).  visible(lb1gt_1,406). happensAt(active(lb1gt_1),406). orientation(lb1gt_2,144,406). coords(lb1gt_2,189,141,406).  visible(lb1gt_2,406). happensAt(active(lb1gt_2),406).
orientation(lb1gt_1,151,407). coords(lb1gt_1,166,168,407).  visible(lb1gt_1,407). happensAt(active(lb1gt_1),407). orientation(lb1gt_2,144,407). coords(lb1gt_2,189,141,407).  visible(lb1gt_2,407). happensAt(active(lb1gt_2),407).
orientation(lb1gt_1,151,408). coords(lb1gt_1,166,168,408).  visible(lb1gt_1,408). happensAt(active(lb1gt_1),408). orientation(lb1gt_2,144,408). coords(lb1gt_2,189,141,408).  visible(lb1gt_2,408). happensAt(active(lb1gt_2),408).
orientation(lb1gt_1,151,409). coords(lb1gt_1,166,168,409).  visible(lb1gt_1,409). happensAt(active(lb1gt_1),409). orientation(lb1gt_2,144,409). coords(lb1gt_2,189,141,409).  visible(lb1gt_2,409). happensAt(active(lb1gt_2),409).
orientation(lb1gt_1,151,410). coords(lb1gt_1,166,168,410).  visible(lb1gt_1,410). happensAt(active(lb1gt_1),410). orientation(lb1gt_2,144,410). coords(lb1gt_2,189,141,410).  visible(lb1gt_2,410). happensAt(active(lb1gt_2),410).
orientation(lb1gt_1,151,411). coords(lb1gt_1,167,168,411).  visible(lb1gt_1,411). happensAt(active(lb1gt_1),411). orientation(lb1gt_2,144,411). coords(lb1gt_2,189,141,411).  visible(lb1gt_2,411). happensAt(active(lb1gt_2),411).
orientation(lb1gt_1,151,412). coords(lb1gt_1,167,168,412).  visible(lb1gt_1,412). happensAt(active(lb1gt_1),412). orientation(lb1gt_2,144,412). coords(lb1gt_2,189,141,412).  visible(lb1gt_2,412). happensAt(active(lb1gt_2),412).
orientation(lb1gt_1,151,413). coords(lb1gt_1,167,168,413).  visible(lb1gt_1,413). happensAt(active(lb1gt_1),413). orientation(lb1gt_2,144,413). coords(lb1gt_2,189,141,413).  visible(lb1gt_2,413). happensAt(active(lb1gt_2),413).
orientation(lb1gt_1,151,414). coords(lb1gt_1,167,170,414).  visible(lb1gt_1,414). happensAt(active(lb1gt_1),414). orientation(lb1gt_2,144,414). coords(lb1gt_2,189,141,414).  visible(lb1gt_2,414). happensAt(active(lb1gt_2),414).
orientation(lb1gt_1,151,415). coords(lb1gt_1,167,170,415).  visible(lb1gt_1,415). happensAt(active(lb1gt_1),415). orientation(lb1gt_2,144,415). coords(lb1gt_2,189,141,415).  visible(lb1gt_2,415). happensAt(active(lb1gt_2),415).
orientation(lb1gt_1,151,416). coords(lb1gt_1,167,172,416).  visible(lb1gt_1,416). happensAt(active(lb1gt_1),416). orientation(lb1gt_2,144,416). coords(lb1gt_2,189,141,416).  visible(lb1gt_2,416). happensAt(active(lb1gt_2),416).
orientation(lb1gt_1,151,417). coords(lb1gt_1,167,173,417).  visible(lb1gt_1,417). happensAt(active(lb1gt_1),417). orientation(lb1gt_2,144,417). coords(lb1gt_2,189,141,417).  visible(lb1gt_2,417). happensAt(active(lb1gt_2),417).
orientation(lb1gt_1,151,418). coords(lb1gt_1,167,173,418).  visible(lb1gt_1,418). happensAt(active(lb1gt_1),418). orientation(lb1gt_2,144,418). coords(lb1gt_2,189,141,418).  visible(lb1gt_2,418). happensAt(active(lb1gt_2),418).
orientation(lb1gt_1,151,419). coords(lb1gt_1,166,174,419).  visible(lb1gt_1,419). happensAt(active(lb1gt_1),419). orientation(lb1gt_2,144,419). coords(lb1gt_2,189,141,419).  visible(lb1gt_2,419). happensAt(active(lb1gt_2),419).
orientation(lb1gt_1,151,420). coords(lb1gt_1,166,175,420).  visible(lb1gt_1,420). happensAt(walking(lb1gt_1),420). orientation(lb1gt_2,144,420). coords(lb1gt_2,189,141,420).  visible(lb1gt_2,420). happensAt(active(lb1gt_2),420).
#show holdsAt/2.