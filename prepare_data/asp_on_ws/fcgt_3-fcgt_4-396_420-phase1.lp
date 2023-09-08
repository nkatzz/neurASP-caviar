
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

	orientation(fcgt_3,164,396). coords(fcgt_3,100,193,396).  visible(fcgt_3,396). happensAt(walking(fcgt_3),396). orientation(fcgt_4,166,396). coords(fcgt_4,82,206,396).  visible(fcgt_4,396). happensAt(inactive(fcgt_4),396).
orientation(fcgt_3,164,397). coords(fcgt_3,100,194,397).  visible(fcgt_3,397). happensAt(walking(fcgt_3),397). orientation(fcgt_4,166,397). coords(fcgt_4,82,206,397).  visible(fcgt_4,397). happensAt(inactive(fcgt_4),397).
orientation(fcgt_3,164,398). coords(fcgt_3,100,195,398).  visible(fcgt_3,398). happensAt(walking(fcgt_3),398). orientation(fcgt_4,166,398). coords(fcgt_4,82,206,398).  visible(fcgt_4,398). happensAt(inactive(fcgt_4),398).
orientation(fcgt_3,164,399). coords(fcgt_3,100,195,399).  visible(fcgt_3,399). happensAt(walking(fcgt_3),399). orientation(fcgt_4,166,399). coords(fcgt_4,82,206,399).  visible(fcgt_4,399). happensAt(inactive(fcgt_4),399).
orientation(fcgt_3,164,400). coords(fcgt_3,100,196,400).  visible(fcgt_3,400). happensAt(walking(fcgt_3),400). orientation(fcgt_4,166,400). coords(fcgt_4,82,206,400).  visible(fcgt_4,400). happensAt(inactive(fcgt_4),400).
orientation(fcgt_3,164,401). coords(fcgt_3,100,197,401).  visible(fcgt_3,401). happensAt(walking(fcgt_3),401). orientation(fcgt_4,166,401). coords(fcgt_4,82,206,401).  visible(fcgt_4,401). happensAt(inactive(fcgt_4),401).
orientation(fcgt_3,164,402). coords(fcgt_3,99,197,402).  visible(fcgt_3,402). happensAt(walking(fcgt_3),402). orientation(fcgt_4,166,402). coords(fcgt_4,82,206,402).  visible(fcgt_4,402). happensAt(inactive(fcgt_4),402).
orientation(fcgt_3,164,403). coords(fcgt_3,99,198,403).  visible(fcgt_3,403). happensAt(walking(fcgt_3),403). orientation(fcgt_4,166,403). coords(fcgt_4,82,206,403).  visible(fcgt_4,403). happensAt(inactive(fcgt_4),403).
orientation(fcgt_3,164,404). coords(fcgt_3,99,198,404).  visible(fcgt_3,404). happensAt(walking(fcgt_3),404). orientation(fcgt_4,166,404). coords(fcgt_4,82,206,404).  visible(fcgt_4,404). happensAt(inactive(fcgt_4),404).
orientation(fcgt_3,164,405). coords(fcgt_3,99,198,405).  visible(fcgt_3,405). happensAt(walking(fcgt_3),405). orientation(fcgt_4,166,405). coords(fcgt_4,82,206,405).  visible(fcgt_4,405). happensAt(inactive(fcgt_4),405).
orientation(fcgt_3,164,406). coords(fcgt_3,99,199,406).  visible(fcgt_3,406). happensAt(walking(fcgt_3),406). orientation(fcgt_4,166,406). coords(fcgt_4,82,206,406).  visible(fcgt_4,406). happensAt(inactive(fcgt_4),406).
orientation(fcgt_3,164,407). coords(fcgt_3,99,199,407).  visible(fcgt_3,407). happensAt(walking(fcgt_3),407). orientation(fcgt_4,166,407). coords(fcgt_4,82,206,407).  visible(fcgt_4,407). happensAt(inactive(fcgt_4),407).
orientation(fcgt_3,164,408). coords(fcgt_3,99,200,408).  visible(fcgt_3,408). happensAt(walking(fcgt_3),408). orientation(fcgt_4,166,408). coords(fcgt_4,82,206,408).  visible(fcgt_4,408). happensAt(inactive(fcgt_4),408).
orientation(fcgt_3,164,409). coords(fcgt_3,99,200,409).  visible(fcgt_3,409). happensAt(walking(fcgt_3),409). orientation(fcgt_4,166,409). coords(fcgt_4,82,206,409).  visible(fcgt_4,409). happensAt(inactive(fcgt_4),409).
orientation(fcgt_3,164,410). coords(fcgt_3,99,200,410).  visible(fcgt_3,410). happensAt(walking(fcgt_3),410). orientation(fcgt_4,166,410). coords(fcgt_4,82,206,410).  visible(fcgt_4,410). happensAt(inactive(fcgt_4),410).
orientation(fcgt_3,164,411). coords(fcgt_3,99,200,411).  visible(fcgt_3,411). happensAt(walking(fcgt_3),411). orientation(fcgt_4,166,411). coords(fcgt_4,82,206,411).  visible(fcgt_4,411). happensAt(inactive(fcgt_4),411).
orientation(fcgt_3,164,412). coords(fcgt_3,98,200,412).  visible(fcgt_3,412). happensAt(walking(fcgt_3),412). orientation(fcgt_4,166,412). coords(fcgt_4,82,206,412).  visible(fcgt_4,412). happensAt(inactive(fcgt_4),412).
orientation(fcgt_3,164,413). coords(fcgt_3,98,200,413).  visible(fcgt_3,413). happensAt(walking(fcgt_3),413). orientation(fcgt_4,166,413). coords(fcgt_4,82,206,413).  visible(fcgt_4,413). happensAt(inactive(fcgt_4),413).
orientation(fcgt_3,164,414). coords(fcgt_3,98,200,414).  visible(fcgt_3,414). happensAt(walking(fcgt_3),414). orientation(fcgt_4,166,414). coords(fcgt_4,82,206,414).  visible(fcgt_4,414). happensAt(inactive(fcgt_4),414).
orientation(fcgt_3,164,415). coords(fcgt_3,98,201,415).  visible(fcgt_3,415). happensAt(walking(fcgt_3),415). orientation(fcgt_4,166,415). coords(fcgt_4,82,206,415).  visible(fcgt_4,415). happensAt(inactive(fcgt_4),415).
orientation(fcgt_3,164,416). coords(fcgt_3,98,201,416).  visible(fcgt_3,416). happensAt(walking(fcgt_3),416). orientation(fcgt_4,166,416). coords(fcgt_4,82,206,416).  visible(fcgt_4,416). happensAt(inactive(fcgt_4),416).
orientation(fcgt_3,164,417). coords(fcgt_3,98,201,417).  visible(fcgt_3,417). happensAt(walking(fcgt_3),417). orientation(fcgt_4,166,417). coords(fcgt_4,82,206,417).  visible(fcgt_4,417). happensAt(inactive(fcgt_4),417).
orientation(fcgt_3,164,418). coords(fcgt_3,98,201,418).  visible(fcgt_3,418). happensAt(walking(fcgt_3),418). orientation(fcgt_4,166,418). coords(fcgt_4,82,206,418).  visible(fcgt_4,418). happensAt(inactive(fcgt_4),418).
orientation(fcgt_3,164,419). coords(fcgt_3,98,201,419).  visible(fcgt_3,419). happensAt(walking(fcgt_3),419). orientation(fcgt_4,166,419). coords(fcgt_4,82,206,419).  visible(fcgt_4,419). happensAt(inactive(fcgt_4),419).
orientation(fcgt_3,164,420). coords(fcgt_3,98,201,420).  visible(fcgt_3,420). happensAt(walking(fcgt_3),420). orientation(fcgt_4,166,420). coords(fcgt_4,82,206,420).  visible(fcgt_4,420). happensAt(inactive(fcgt_4),420).
#show holdsAt/2.