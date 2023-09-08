
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

	orientation(wk1gt_1,145,404). coords(wk1gt_1,103,107,404).  visible(wk1gt_1,404). happensAt(walking(wk1gt_1),404). orientation(wk1gt_4,160,404). coords(wk1gt_4,52,176,404).  visible(wk1gt_4,404). happensAt(active(wk1gt_4),404).
orientation(wk1gt_1,146,405). coords(wk1gt_1,103,107,405).  visible(wk1gt_1,405). happensAt(walking(wk1gt_1),405). orientation(wk1gt_4,154,405). coords(wk1gt_4,52,176,405).  visible(wk1gt_4,405). happensAt(active(wk1gt_4),405).
orientation(wk1gt_1,145,406). coords(wk1gt_1,103,107,406).  visible(wk1gt_1,406). happensAt(walking(wk1gt_1),406). orientation(wk1gt_4,156,406). coords(wk1gt_4,52,176,406).  visible(wk1gt_4,406). happensAt(active(wk1gt_4),406).
orientation(wk1gt_1,143,407). coords(wk1gt_1,102,106,407).  visible(wk1gt_1,407). happensAt(walking(wk1gt_1),407). orientation(wk1gt_4,159,407). coords(wk1gt_4,52,176,407).  visible(wk1gt_4,407). happensAt(active(wk1gt_4),407).
orientation(wk1gt_1,146,408). coords(wk1gt_1,100,105,408).  visible(wk1gt_1,408). happensAt(walking(wk1gt_1),408). orientation(wk1gt_4,156,408). coords(wk1gt_4,52,176,408).  visible(wk1gt_4,408). happensAt(active(wk1gt_4),408).
orientation(wk1gt_1,146,409). coords(wk1gt_1,100,105,409).  visible(wk1gt_1,409). happensAt(walking(wk1gt_1),409). orientation(wk1gt_4,155,409). coords(wk1gt_4,52,176,409).  visible(wk1gt_4,409). happensAt(active(wk1gt_4),409).
orientation(wk1gt_1,146,410). coords(wk1gt_1,99,104,410).  visible(wk1gt_1,410). happensAt(walking(wk1gt_1),410). orientation(wk1gt_4,162,410). coords(wk1gt_4,52,176,410).  visible(wk1gt_4,410). happensAt(active(wk1gt_4),410).
orientation(wk1gt_1,146,411). coords(wk1gt_1,99,103,411).  visible(wk1gt_1,411). happensAt(walking(wk1gt_1),411). orientation(wk1gt_4,168,411). coords(wk1gt_4,52,176,411).  visible(wk1gt_4,411). happensAt(active(wk1gt_4),411).
orientation(wk1gt_1,146,412). coords(wk1gt_1,99,103,412).  visible(wk1gt_1,412). happensAt(walking(wk1gt_1),412). orientation(wk1gt_4,162,412). coords(wk1gt_4,52,176,412).  visible(wk1gt_4,412). happensAt(active(wk1gt_4),412).
orientation(wk1gt_1,146,413). coords(wk1gt_1,99,102,413).  visible(wk1gt_1,413). happensAt(walking(wk1gt_1),413). orientation(wk1gt_4,162,413). coords(wk1gt_4,52,176,413).  visible(wk1gt_4,413). happensAt(active(wk1gt_4),413).
orientation(wk1gt_1,146,414). coords(wk1gt_1,99,102,414).  visible(wk1gt_1,414). happensAt(walking(wk1gt_1),414). orientation(wk1gt_4,161,414). coords(wk1gt_4,52,176,414).  visible(wk1gt_4,414). happensAt(active(wk1gt_4),414).
orientation(wk1gt_1,146,415). coords(wk1gt_1,98,101,415).  visible(wk1gt_1,415). happensAt(walking(wk1gt_1),415). orientation(wk1gt_4,163,415). coords(wk1gt_4,52,176,415).  visible(wk1gt_4,415). happensAt(active(wk1gt_4),415).
orientation(wk1gt_1,146,416). coords(wk1gt_1,98,101,416).  visible(wk1gt_1,416). happensAt(walking(wk1gt_1),416). orientation(wk1gt_4,163,416). coords(wk1gt_4,53,176,416).  visible(wk1gt_4,416). happensAt(active(wk1gt_4),416).
orientation(wk1gt_1,146,417). coords(wk1gt_1,98,101,417).  visible(wk1gt_1,417). happensAt(walking(wk1gt_1),417). orientation(wk1gt_4,160,417). coords(wk1gt_4,53,175,417).  visible(wk1gt_4,417). happensAt(active(wk1gt_4),417).
orientation(wk1gt_1,146,418). coords(wk1gt_1,97,101,418).  visible(wk1gt_1,418). happensAt(walking(wk1gt_1),418). orientation(wk1gt_4,159,418). coords(wk1gt_4,53,175,418).  visible(wk1gt_4,418). happensAt(active(wk1gt_4),418).
orientation(wk1gt_1,146,419). coords(wk1gt_1,97,100,419).  visible(wk1gt_1,419). happensAt(walking(wk1gt_1),419). orientation(wk1gt_4,157,419). coords(wk1gt_4,54,175,419).  visible(wk1gt_4,419). happensAt(active(wk1gt_4),419).
orientation(wk1gt_1,146,420). coords(wk1gt_1,97,100,420).  visible(wk1gt_1,420). happensAt(walking(wk1gt_1),420). orientation(wk1gt_4,157,420). coords(wk1gt_4,54,175,420).  visible(wk1gt_4,420). happensAt(active(wk1gt_4),420).
orientation(wk1gt_1,146,421). coords(wk1gt_1,96,99,421).  visible(wk1gt_1,421). happensAt(walking(wk1gt_1),421). orientation(wk1gt_4,157,421). coords(wk1gt_4,54,175,421).  visible(wk1gt_4,421). happensAt(active(wk1gt_4),421).
orientation(wk1gt_1,146,422). coords(wk1gt_1,95,98,422).  visible(wk1gt_1,422). happensAt(walking(wk1gt_1),422). orientation(wk1gt_4,157,422). coords(wk1gt_4,54,174,422).  visible(wk1gt_4,422). happensAt(active(wk1gt_4),422).
orientation(wk1gt_1,153,423). coords(wk1gt_1,94,98,423).  visible(wk1gt_1,423). happensAt(walking(wk1gt_1),423). orientation(wk1gt_4,157,423). coords(wk1gt_4,54,174,423).  visible(wk1gt_4,423). happensAt(active(wk1gt_4),423).
orientation(wk1gt_1,153,424). coords(wk1gt_1,94,98,424).  visible(wk1gt_1,424). happensAt(walking(wk1gt_1),424). orientation(wk1gt_4,157,424). coords(wk1gt_4,54,174,424).  visible(wk1gt_4,424). happensAt(active(wk1gt_4),424).
orientation(wk1gt_1,153,425). coords(wk1gt_1,94,98,425).  visible(wk1gt_1,425). happensAt(walking(wk1gt_1),425). orientation(wk1gt_4,157,425). coords(wk1gt_4,54,173,425).  visible(wk1gt_4,425). happensAt(active(wk1gt_4),425).
orientation(wk1gt_1,153,426). coords(wk1gt_1,94,98,426).  visible(wk1gt_1,426). happensAt(walking(wk1gt_1),426). orientation(wk1gt_4,147,426). coords(wk1gt_4,54,173,426).  visible(wk1gt_4,426). happensAt(active(wk1gt_4),426).
orientation(wk1gt_1,150,427). coords(wk1gt_1,94,98,427).  visible(wk1gt_1,427). happensAt(walking(wk1gt_1),427). orientation(wk1gt_4,147,427). coords(wk1gt_4,54,172,427).  visible(wk1gt_4,427). happensAt(active(wk1gt_4),427).
orientation(wk1gt_1,150,428). coords(wk1gt_1,94,98,428).  visible(wk1gt_1,428). happensAt(walking(wk1gt_1),428). orientation(wk1gt_4,147,428). coords(wk1gt_4,54,172,428).  visible(wk1gt_4,428). happensAt(active(wk1gt_4),428).
#show holdsAt/2.