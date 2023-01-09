
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

	orientation(fra2gt_3,164,406). coords(fra2gt_3,62,145,406).  visible(fra2gt_3,406). happensAt(walking(fra2gt_3),406). orientation(fra2gt_4,138,406). coords(fra2gt_4,75,131,406).  visible(fra2gt_4,406). happensAt(running(fra2gt_4),406).
orientation(fra2gt_3,164,407). coords(fra2gt_3,62,145,407).  visible(fra2gt_3,407). happensAt(walking(fra2gt_3),407). orientation(fra2gt_4,138,407). coords(fra2gt_4,75,131,407).  visible(fra2gt_4,407). happensAt(running(fra2gt_4),407).
orientation(fra2gt_3,164,408). coords(fra2gt_3,62,145,408).  visible(fra2gt_3,408). happensAt(walking(fra2gt_3),408). orientation(fra2gt_4,147,408). coords(fra2gt_4,75,129,408).  visible(fra2gt_4,408). happensAt(running(fra2gt_4),408).
orientation(fra2gt_3,164,409). coords(fra2gt_3,61,146,409).  visible(fra2gt_3,409). happensAt(walking(fra2gt_3),409). orientation(fra2gt_4,147,409). coords(fra2gt_4,75,127,409).  visible(fra2gt_4,409). happensAt(running(fra2gt_4),409).
orientation(fra2gt_3,164,410). coords(fra2gt_3,60,146,410).  visible(fra2gt_3,410). happensAt(walking(fra2gt_3),410). orientation(fra2gt_4,147,410). coords(fra2gt_4,75,125,410).  visible(fra2gt_4,410). happensAt(running(fra2gt_4),410).
orientation(fra2gt_3,164,411). coords(fra2gt_3,59,148,411).  visible(fra2gt_3,411). happensAt(walking(fra2gt_3),411). orientation(fra2gt_4,147,411). coords(fra2gt_4,74,125,411).  visible(fra2gt_4,411). happensAt(running(fra2gt_4),411).
orientation(fra2gt_3,164,412). coords(fra2gt_3,59,149,412).  visible(fra2gt_3,412). happensAt(walking(fra2gt_3),412). orientation(fra2gt_4,147,412). coords(fra2gt_4,74,124,412).  visible(fra2gt_4,412). happensAt(running(fra2gt_4),412).
orientation(fra2gt_3,164,413). coords(fra2gt_3,58,150,413).  visible(fra2gt_3,413). happensAt(walking(fra2gt_3),413). orientation(fra2gt_4,147,413). coords(fra2gt_4,74,123,413).  visible(fra2gt_4,413). happensAt(running(fra2gt_4),413).
orientation(fra2gt_3,164,414). coords(fra2gt_3,58,152,414).  visible(fra2gt_3,414). happensAt(walking(fra2gt_3),414). orientation(fra2gt_4,147,414). coords(fra2gt_4,74,123,414).  visible(fra2gt_4,414). happensAt(running(fra2gt_4),414).
orientation(fra2gt_3,164,415). coords(fra2gt_3,58,153,415).  visible(fra2gt_3,415). happensAt(walking(fra2gt_3),415). orientation(fra2gt_4,137,415). coords(fra2gt_4,74,122,415).  visible(fra2gt_4,415). happensAt(running(fra2gt_4),415).
orientation(fra2gt_3,164,416). coords(fra2gt_3,57,153,416).  visible(fra2gt_3,416). happensAt(walking(fra2gt_3),416). orientation(fra2gt_4,137,416). coords(fra2gt_4,74,120,416).  visible(fra2gt_4,416). happensAt(running(fra2gt_4),416).
orientation(fra2gt_3,164,417). coords(fra2gt_3,57,153,417).  visible(fra2gt_3,417). happensAt(walking(fra2gt_3),417). orientation(fra2gt_4,137,417). coords(fra2gt_4,74,119,417).  visible(fra2gt_4,417). happensAt(running(fra2gt_4),417).
orientation(fra2gt_3,164,418). coords(fra2gt_3,57,154,418).  visible(fra2gt_3,418). happensAt(walking(fra2gt_3),418). orientation(fra2gt_4,137,418). coords(fra2gt_4,74,118,418).  visible(fra2gt_4,418). happensAt(running(fra2gt_4),418).
orientation(fra2gt_3,164,419). coords(fra2gt_3,56,155,419).  visible(fra2gt_3,419). happensAt(walking(fra2gt_3),419). orientation(fra2gt_4,137,419). coords(fra2gt_4,74,116,419).  visible(fra2gt_4,419). happensAt(running(fra2gt_4),419).
orientation(fra2gt_3,164,420). coords(fra2gt_3,56,155,420).  visible(fra2gt_3,420). happensAt(walking(fra2gt_3),420). orientation(fra2gt_4,137,420). coords(fra2gt_4,74,114,420).  visible(fra2gt_4,420). happensAt(running(fra2gt_4),420).
orientation(fra2gt_3,164,421). coords(fra2gt_3,55,156,421).  visible(fra2gt_3,421). happensAt(walking(fra2gt_3),421). orientation(fra2gt_4,137,421). coords(fra2gt_4,73,112,421).  visible(fra2gt_4,421). happensAt(running(fra2gt_4),421).
orientation(fra2gt_3,164,422). coords(fra2gt_3,55,156,422).  visible(fra2gt_3,422). happensAt(walking(fra2gt_3),422). orientation(fra2gt_4,137,422). coords(fra2gt_4,73,110,422).  visible(fra2gt_4,422). happensAt(running(fra2gt_4),422).
orientation(fra2gt_3,164,423). coords(fra2gt_3,55,157,423).  visible(fra2gt_3,423). happensAt(walking(fra2gt_3),423). orientation(fra2gt_4,137,423). coords(fra2gt_4,73,109,423).  visible(fra2gt_4,423). happensAt(running(fra2gt_4),423).
orientation(fra2gt_3,164,424). coords(fra2gt_3,54,159,424).  visible(fra2gt_3,424). happensAt(walking(fra2gt_3),424). orientation(fra2gt_4,137,424). coords(fra2gt_4,72,109,424).  visible(fra2gt_4,424). happensAt(running(fra2gt_4),424).
orientation(fra2gt_3,164,425). coords(fra2gt_3,55,159,425).  visible(fra2gt_3,425). happensAt(walking(fra2gt_3),425). orientation(fra2gt_4,141,425). coords(fra2gt_4,72,107,425).  visible(fra2gt_4,425). happensAt(running(fra2gt_4),425).
orientation(fra2gt_3,164,426). coords(fra2gt_3,54,159,426).  visible(fra2gt_3,426). happensAt(walking(fra2gt_3),426). orientation(fra2gt_4,141,426). coords(fra2gt_4,72,107,426).  visible(fra2gt_4,426). happensAt(running(fra2gt_4),426).
orientation(fra2gt_3,164,427). coords(fra2gt_3,54,160,427).  visible(fra2gt_3,427). happensAt(walking(fra2gt_3),427). orientation(fra2gt_4,141,427). coords(fra2gt_4,72,105,427).  visible(fra2gt_4,427). happensAt(running(fra2gt_4),427).
orientation(fra2gt_3,164,428). coords(fra2gt_3,53,161,428).  visible(fra2gt_3,428). happensAt(walking(fra2gt_3),428). orientation(fra2gt_4,141,428). coords(fra2gt_4,72,104,428).  visible(fra2gt_4,428). happensAt(running(fra2gt_4),428).
orientation(fra2gt_3,164,429). coords(fra2gt_3,52,161,429).  visible(fra2gt_3,429). happensAt(walking(fra2gt_3),429). orientation(fra2gt_4,141,429). coords(fra2gt_4,72,104,429).  visible(fra2gt_4,429). happensAt(running(fra2gt_4),429).
orientation(fra2gt_3,164,430). coords(fra2gt_3,52,161,430).  visible(fra2gt_3,430). happensAt(walking(fra2gt_3),430). orientation(fra2gt_4,146,430). coords(fra2gt_4,73,104,430).  visible(fra2gt_4,430). happensAt(running(fra2gt_4),430).
#show holdsAt/2.