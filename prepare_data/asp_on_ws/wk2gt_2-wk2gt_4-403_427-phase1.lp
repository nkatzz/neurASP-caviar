
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
orientation(wk2gt_2,141,416). coords(wk2gt_2,98,63,416).  visible(wk2gt_2,416). happensAt(inactive(wk2gt_2),416). orientation(wk2gt_4,149,416). coords(wk2gt_4,116,106,416).  visible(wk2gt_4,416). happensAt(walking(wk2gt_4),416).
orientation(wk2gt_2,141,417). coords(wk2gt_2,98,63,417).  visible(wk2gt_2,417). happensAt(inactive(wk2gt_2),417). orientation(wk2gt_4,149,417). coords(wk2gt_4,114,106,417).  visible(wk2gt_4,417). happensAt(walking(wk2gt_4),417).
orientation(wk2gt_2,141,418). coords(wk2gt_2,98,63,418).  visible(wk2gt_2,418). happensAt(inactive(wk2gt_2),418). orientation(wk2gt_4,149,418). coords(wk2gt_4,113,106,418).  visible(wk2gt_4,418). happensAt(walking(wk2gt_4),418).
orientation(wk2gt_2,141,419). coords(wk2gt_2,98,63,419).  visible(wk2gt_2,419). happensAt(inactive(wk2gt_2),419). orientation(wk2gt_4,149,419). coords(wk2gt_4,113,106,419).  visible(wk2gt_4,419). happensAt(walking(wk2gt_4),419).
orientation(wk2gt_2,141,420). coords(wk2gt_2,98,63,420).  visible(wk2gt_2,420). happensAt(inactive(wk2gt_2),420). orientation(wk2gt_4,149,420). coords(wk2gt_4,113,105,420).  visible(wk2gt_4,420). happensAt(walking(wk2gt_4),420).
orientation(wk2gt_2,141,421). coords(wk2gt_2,98,63,421).  visible(wk2gt_2,421). happensAt(inactive(wk2gt_2),421). orientation(wk2gt_4,149,421). coords(wk2gt_4,113,105,421).  visible(wk2gt_4,421). happensAt(walking(wk2gt_4),421).
orientation(wk2gt_2,141,422). coords(wk2gt_2,98,63,422).  visible(wk2gt_2,422). happensAt(inactive(wk2gt_2),422). orientation(wk2gt_4,149,422). coords(wk2gt_4,113,105,422).  visible(wk2gt_4,422). happensAt(walking(wk2gt_4),422).
orientation(wk2gt_2,141,423). coords(wk2gt_2,98,63,423).  visible(wk2gt_2,423). happensAt(inactive(wk2gt_2),423). orientation(wk2gt_4,149,423). coords(wk2gt_4,112,105,423).  visible(wk2gt_4,423). happensAt(walking(wk2gt_4),423).
orientation(wk2gt_2,141,424). coords(wk2gt_2,98,63,424).  visible(wk2gt_2,424). happensAt(inactive(wk2gt_2),424). orientation(wk2gt_4,149,424). coords(wk2gt_4,111,105,424).  visible(wk2gt_4,424). happensAt(walking(wk2gt_4),424).
orientation(wk2gt_2,141,425). coords(wk2gt_2,98,63,425).  visible(wk2gt_2,425). happensAt(inactive(wk2gt_2),425). orientation(wk2gt_4,149,425). coords(wk2gt_4,110,104,425).  visible(wk2gt_4,425). happensAt(walking(wk2gt_4),425).
orientation(wk2gt_2,141,426). coords(wk2gt_2,98,63,426).  visible(wk2gt_2,426). happensAt(inactive(wk2gt_2),426). orientation(wk2gt_4,149,426). coords(wk2gt_4,109,104,426).  visible(wk2gt_4,426). happensAt(walking(wk2gt_4),426).
orientation(wk2gt_2,141,427). coords(wk2gt_2,98,63,427).  visible(wk2gt_2,427). happensAt(inactive(wk2gt_2),427). orientation(wk2gt_4,149,427). coords(wk2gt_4,107,103,427).  visible(wk2gt_4,427). happensAt(walking(wk2gt_4),427).
#show holdsAt/2.