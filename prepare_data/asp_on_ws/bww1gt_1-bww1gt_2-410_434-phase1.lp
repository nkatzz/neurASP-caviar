
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
orientation(bww1gt_1,2,423). coords(bww1gt_1,105,264,423).  visible(bww1gt_1,423). happensAt(inactive(bww1gt_1),423). orientation(bww1gt_2,156,423). coords(bww1gt_2,133,145,423).  visible(bww1gt_2,423). happensAt(walking(bww1gt_2),423).
orientation(bww1gt_1,3,424). coords(bww1gt_1,105,264,424).  visible(bww1gt_1,424). happensAt(inactive(bww1gt_1),424). orientation(bww1gt_2,155,424). coords(bww1gt_2,134,144,424).  visible(bww1gt_2,424). happensAt(walking(bww1gt_2),424).
orientation(bww1gt_1,1,425). coords(bww1gt_1,105,264,425).  visible(bww1gt_1,425). happensAt(inactive(bww1gt_1),425). orientation(bww1gt_2,153,425). coords(bww1gt_2,134,144,425).  visible(bww1gt_2,425). happensAt(walking(bww1gt_2),425).
orientation(bww1gt_1,2,426). coords(bww1gt_1,105,264,426).  visible(bww1gt_1,426). happensAt(inactive(bww1gt_1),426). orientation(bww1gt_2,152,426). coords(bww1gt_2,134,144,426).  visible(bww1gt_2,426). happensAt(walking(bww1gt_2),426).
orientation(bww1gt_1,179,427). coords(bww1gt_1,105,264,427).  visible(bww1gt_1,427). happensAt(inactive(bww1gt_1),427). orientation(bww1gt_2,153,427). coords(bww1gt_2,134,144,427).  visible(bww1gt_2,427). happensAt(walking(bww1gt_2),427).
orientation(bww1gt_1,3,428). coords(bww1gt_1,105,264,428).  visible(bww1gt_1,428). happensAt(inactive(bww1gt_1),428). orientation(bww1gt_2,152,428). coords(bww1gt_2,135,144,428).  visible(bww1gt_2,428). happensAt(walking(bww1gt_2),428).
orientation(bww1gt_1,4,429). coords(bww1gt_1,105,264,429).  visible(bww1gt_1,429). happensAt(inactive(bww1gt_1),429). orientation(bww1gt_2,151,429). coords(bww1gt_2,135,144,429).  visible(bww1gt_2,429). happensAt(walking(bww1gt_2),429).
orientation(bww1gt_1,5,430). coords(bww1gt_1,105,264,430).  visible(bww1gt_1,430). happensAt(inactive(bww1gt_1),430). orientation(bww1gt_2,153,430). coords(bww1gt_2,136,144,430).  visible(bww1gt_2,430). happensAt(walking(bww1gt_2),430).
orientation(bww1gt_1,1,431). coords(bww1gt_1,105,264,431).  visible(bww1gt_1,431). happensAt(inactive(bww1gt_1),431). orientation(bww1gt_2,155,431). coords(bww1gt_2,138,145,431).  visible(bww1gt_2,431). happensAt(walking(bww1gt_2),431).
orientation(bww1gt_1,7,432). coords(bww1gt_1,105,264,432).  visible(bww1gt_1,432). happensAt(inactive(bww1gt_1),432). orientation(bww1gt_2,153,432). coords(bww1gt_2,140,145,432).  visible(bww1gt_2,432). happensAt(walking(bww1gt_2),432).
orientation(bww1gt_1,2,433). coords(bww1gt_1,105,264,433).  visible(bww1gt_1,433). happensAt(inactive(bww1gt_1),433). orientation(bww1gt_2,149,433). coords(bww1gt_2,141,145,433).  visible(bww1gt_2,433). happensAt(walking(bww1gt_2),433).
orientation(bww1gt_1,178,434). coords(bww1gt_1,105,264,434).  visible(bww1gt_1,434). happensAt(inactive(bww1gt_1),434). orientation(bww1gt_2,151,434). coords(bww1gt_2,142,146,434).  visible(bww1gt_2,434). happensAt(walking(bww1gt_2),434).
#show holdsAt/2.