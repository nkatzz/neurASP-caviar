
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

	orientation(wk2gt_1,125,403). coords(wk2gt_1,115,53,403).  visible(wk2gt_1,403). happensAt(inactive(wk2gt_1),403). orientation(wk2gt_4,146,403). coords(wk2gt_4,128,112,403).  visible(wk2gt_4,403). happensAt(walking(wk2gt_4),403).
orientation(wk2gt_1,125,404). coords(wk2gt_1,115,53,404).  visible(wk2gt_1,404). happensAt(inactive(wk2gt_1),404). orientation(wk2gt_4,146,404). coords(wk2gt_4,127,110,404).  visible(wk2gt_4,404). happensAt(walking(wk2gt_4),404).
orientation(wk2gt_1,125,405). coords(wk2gt_1,115,53,405).  visible(wk2gt_1,405). happensAt(inactive(wk2gt_1),405). orientation(wk2gt_4,146,405). coords(wk2gt_4,126,110,405).  visible(wk2gt_4,405). happensAt(walking(wk2gt_4),405).
orientation(wk2gt_1,125,406). coords(wk2gt_1,115,53,406).  visible(wk2gt_1,406). happensAt(inactive(wk2gt_1),406). orientation(wk2gt_4,146,406). coords(wk2gt_4,125,109,406).  visible(wk2gt_4,406). happensAt(walking(wk2gt_4),406).
orientation(wk2gt_1,125,407). coords(wk2gt_1,115,53,407).  visible(wk2gt_1,407). happensAt(inactive(wk2gt_1),407). orientation(wk2gt_4,149,407). coords(wk2gt_4,125,108,407).  visible(wk2gt_4,407). happensAt(walking(wk2gt_4),407).
orientation(wk2gt_1,125,408). coords(wk2gt_1,115,53,408).  visible(wk2gt_1,408). happensAt(inactive(wk2gt_1),408). orientation(wk2gt_4,149,408). coords(wk2gt_4,124,108,408).  visible(wk2gt_4,408). happensAt(walking(wk2gt_4),408).
orientation(wk2gt_1,125,409). coords(wk2gt_1,115,53,409).  visible(wk2gt_1,409). happensAt(inactive(wk2gt_1),409). orientation(wk2gt_4,149,409). coords(wk2gt_4,124,108,409).  visible(wk2gt_4,409). happensAt(walking(wk2gt_4),409).
orientation(wk2gt_1,125,410). coords(wk2gt_1,115,53,410).  visible(wk2gt_1,410). happensAt(inactive(wk2gt_1),410). orientation(wk2gt_4,149,410). coords(wk2gt_4,124,108,410).  visible(wk2gt_4,410). happensAt(walking(wk2gt_4),410).
orientation(wk2gt_1,125,411). coords(wk2gt_1,115,53,411).  visible(wk2gt_1,411). happensAt(inactive(wk2gt_1),411). orientation(wk2gt_4,149,411). coords(wk2gt_4,123,107,411).  visible(wk2gt_4,411). happensAt(walking(wk2gt_4),411).
orientation(wk2gt_1,125,412). coords(wk2gt_1,115,53,412).  visible(wk2gt_1,412). happensAt(inactive(wk2gt_1),412). orientation(wk2gt_4,149,412). coords(wk2gt_4,122,107,412).  visible(wk2gt_4,412). happensAt(walking(wk2gt_4),412).
orientation(wk2gt_1,125,413). coords(wk2gt_1,115,53,413).  visible(wk2gt_1,413). happensAt(inactive(wk2gt_1),413). orientation(wk2gt_4,149,413). coords(wk2gt_4,121,107,413).  visible(wk2gt_4,413). happensAt(walking(wk2gt_4),413).
orientation(wk2gt_1,125,414). coords(wk2gt_1,115,53,414).  visible(wk2gt_1,414). happensAt(inactive(wk2gt_1),414). orientation(wk2gt_4,149,414). coords(wk2gt_4,120,107,414).  visible(wk2gt_4,414). happensAt(walking(wk2gt_4),414).
orientation(wk2gt_1,125,415). coords(wk2gt_1,115,53,415).  visible(wk2gt_1,415). happensAt(inactive(wk2gt_1),415). orientation(wk2gt_4,149,415). coords(wk2gt_4,118,106,415).  visible(wk2gt_4,415). happensAt(walking(wk2gt_4),415).
orientation(wk2gt_1,125,416). coords(wk2gt_1,115,53,416).  visible(wk2gt_1,416). happensAt(inactive(wk2gt_1),416). orientation(wk2gt_4,149,416). coords(wk2gt_4,116,106,416).  visible(wk2gt_4,416). happensAt(walking(wk2gt_4),416).
orientation(wk2gt_1,125,417). coords(wk2gt_1,115,53,417).  visible(wk2gt_1,417). happensAt(inactive(wk2gt_1),417). orientation(wk2gt_4,149,417). coords(wk2gt_4,114,106,417).  visible(wk2gt_4,417). happensAt(walking(wk2gt_4),417).
orientation(wk2gt_1,125,418). coords(wk2gt_1,115,53,418).  visible(wk2gt_1,418). happensAt(inactive(wk2gt_1),418). orientation(wk2gt_4,149,418). coords(wk2gt_4,113,106,418).  visible(wk2gt_4,418). happensAt(walking(wk2gt_4),418).
orientation(wk2gt_1,125,419). coords(wk2gt_1,115,53,419).  visible(wk2gt_1,419). happensAt(inactive(wk2gt_1),419). orientation(wk2gt_4,149,419). coords(wk2gt_4,113,106,419).  visible(wk2gt_4,419). happensAt(walking(wk2gt_4),419).
orientation(wk2gt_1,125,420). coords(wk2gt_1,115,53,420).  visible(wk2gt_1,420). happensAt(inactive(wk2gt_1),420). orientation(wk2gt_4,149,420). coords(wk2gt_4,113,105,420).  visible(wk2gt_4,420). happensAt(walking(wk2gt_4),420).
orientation(wk2gt_1,125,421). coords(wk2gt_1,115,53,421).  visible(wk2gt_1,421). happensAt(inactive(wk2gt_1),421). orientation(wk2gt_4,149,421). coords(wk2gt_4,113,105,421).  visible(wk2gt_4,421). happensAt(walking(wk2gt_4),421).
orientation(wk2gt_1,125,422). coords(wk2gt_1,115,53,422).  visible(wk2gt_1,422). happensAt(inactive(wk2gt_1),422). orientation(wk2gt_4,149,422). coords(wk2gt_4,113,105,422).  visible(wk2gt_4,422). happensAt(walking(wk2gt_4),422).
orientation(wk2gt_1,125,423). coords(wk2gt_1,115,53,423).  visible(wk2gt_1,423). happensAt(inactive(wk2gt_1),423). orientation(wk2gt_4,149,423). coords(wk2gt_4,112,105,423).  visible(wk2gt_4,423). happensAt(walking(wk2gt_4),423).
orientation(wk2gt_1,125,424). coords(wk2gt_1,115,53,424).  visible(wk2gt_1,424). happensAt(inactive(wk2gt_1),424). orientation(wk2gt_4,149,424). coords(wk2gt_4,111,105,424).  visible(wk2gt_4,424). happensAt(walking(wk2gt_4),424).
orientation(wk2gt_1,125,425). coords(wk2gt_1,115,53,425).  visible(wk2gt_1,425). happensAt(inactive(wk2gt_1),425). orientation(wk2gt_4,149,425). coords(wk2gt_4,110,104,425).  visible(wk2gt_4,425). happensAt(walking(wk2gt_4),425).
orientation(wk2gt_1,125,426). coords(wk2gt_1,115,53,426).  visible(wk2gt_1,426). happensAt(inactive(wk2gt_1),426). orientation(wk2gt_4,149,426). coords(wk2gt_4,109,104,426).  visible(wk2gt_4,426). happensAt(walking(wk2gt_4),426).
orientation(wk2gt_1,125,427). coords(wk2gt_1,115,53,427).  visible(wk2gt_1,427). happensAt(inactive(wk2gt_1),427). orientation(wk2gt_4,149,427). coords(wk2gt_4,107,103,427).  visible(wk2gt_4,427). happensAt(walking(wk2gt_4),427).
#show holdsAt/2.