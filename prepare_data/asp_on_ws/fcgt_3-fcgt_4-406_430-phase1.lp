
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
orientation(fcgt_3,164,421). coords(fcgt_3,98,201,421).  visible(fcgt_3,421). happensAt(walking(fcgt_3),421). orientation(fcgt_4,166,421). coords(fcgt_4,82,206,421).  visible(fcgt_4,421). happensAt(inactive(fcgt_4),421).
orientation(fcgt_3,164,422). coords(fcgt_3,98,201,422).  visible(fcgt_3,422). happensAt(walking(fcgt_3),422). orientation(fcgt_4,166,422). coords(fcgt_4,82,206,422).  visible(fcgt_4,422). happensAt(inactive(fcgt_4),422).
orientation(fcgt_3,164,423). coords(fcgt_3,98,201,423).  visible(fcgt_3,423). happensAt(walking(fcgt_3),423). orientation(fcgt_4,166,423). coords(fcgt_4,82,206,423).  visible(fcgt_4,423). happensAt(inactive(fcgt_4),423).
orientation(fcgt_3,164,424). coords(fcgt_3,98,201,424).  visible(fcgt_3,424). happensAt(walking(fcgt_3),424). orientation(fcgt_4,166,424). coords(fcgt_4,82,206,424).  visible(fcgt_4,424). happensAt(inactive(fcgt_4),424).
orientation(fcgt_3,164,425). coords(fcgt_3,98,201,425).  visible(fcgt_3,425). happensAt(walking(fcgt_3),425). orientation(fcgt_4,166,425). coords(fcgt_4,82,206,425).  visible(fcgt_4,425). happensAt(inactive(fcgt_4),425).
orientation(fcgt_3,164,426). coords(fcgt_3,98,201,426).  visible(fcgt_3,426). happensAt(walking(fcgt_3),426). orientation(fcgt_4,166,426). coords(fcgt_4,82,206,426).  visible(fcgt_4,426). happensAt(inactive(fcgt_4),426).
orientation(fcgt_3,164,427). coords(fcgt_3,98,202,427).  visible(fcgt_3,427). happensAt(walking(fcgt_3),427). orientation(fcgt_4,166,427). coords(fcgt_4,82,206,427).  visible(fcgt_4,427). happensAt(inactive(fcgt_4),427).
orientation(fcgt_3,164,428). coords(fcgt_3,98,202,428).  visible(fcgt_3,428). happensAt(walking(fcgt_3),428). orientation(fcgt_4,166,428). coords(fcgt_4,82,206,428).  visible(fcgt_4,428). happensAt(inactive(fcgt_4),428).
orientation(fcgt_3,164,429). coords(fcgt_3,98,202,429).  visible(fcgt_3,429). happensAt(walking(fcgt_3),429). orientation(fcgt_4,166,429). coords(fcgt_4,82,206,429).  visible(fcgt_4,429). happensAt(inactive(fcgt_4),429).
orientation(fcgt_3,164,430). coords(fcgt_3,98,202,430).  visible(fcgt_3,430). happensAt(walking(fcgt_3),430). orientation(fcgt_4,166,430). coords(fcgt_4,82,206,430).  visible(fcgt_4,430). happensAt(inactive(fcgt_4),430).
#show holdsAt/2.