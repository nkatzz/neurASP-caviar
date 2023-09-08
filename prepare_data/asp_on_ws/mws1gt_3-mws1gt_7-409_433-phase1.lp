
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

	orientation(mws1gt_3,165,409). coords(mws1gt_3,34,160,409).  visible(mws1gt_3,409). happensAt(walking(mws1gt_3),409). orientation(mws1gt_7,174,409). coords(mws1gt_7,67,190,409).  visible(mws1gt_7,409). happensAt(walking(mws1gt_7),409).
orientation(mws1gt_3,165,410). coords(mws1gt_3,34,160,410).  visible(mws1gt_3,410). happensAt(walking(mws1gt_3),410). orientation(mws1gt_7,174,410). coords(mws1gt_7,67,190,410).  visible(mws1gt_7,410). happensAt(walking(mws1gt_7),410).
orientation(mws1gt_3,165,411). coords(mws1gt_3,34,160,411).  visible(mws1gt_3,411). happensAt(walking(mws1gt_3),411). orientation(mws1gt_7,174,411). coords(mws1gt_7,67,190,411).  visible(mws1gt_7,411). happensAt(walking(mws1gt_7),411).
orientation(mws1gt_3,165,412). coords(mws1gt_3,33,160,412).  visible(mws1gt_3,412). happensAt(walking(mws1gt_3),412). orientation(mws1gt_7,174,412). coords(mws1gt_7,67,190,412).  visible(mws1gt_7,412). happensAt(walking(mws1gt_7),412).
orientation(mws1gt_3,165,413). coords(mws1gt_3,33,160,413).  visible(mws1gt_3,413). happensAt(walking(mws1gt_3),413). orientation(mws1gt_7,174,413). coords(mws1gt_7,67,190,413).  visible(mws1gt_7,413). happensAt(walking(mws1gt_7),413).
orientation(mws1gt_3,165,414). coords(mws1gt_3,33,160,414).  visible(mws1gt_3,414). happensAt(walking(mws1gt_3),414). orientation(mws1gt_7,174,414). coords(mws1gt_7,68,191,414).  visible(mws1gt_7,414). happensAt(walking(mws1gt_7),414).
orientation(mws1gt_3,165,415). coords(mws1gt_3,33,160,415).  visible(mws1gt_3,415). happensAt(walking(mws1gt_3),415). orientation(mws1gt_7,174,415). coords(mws1gt_7,69,191,415).  visible(mws1gt_7,415). happensAt(walking(mws1gt_7),415).
orientation(mws1gt_3,165,416). coords(mws1gt_3,33,160,416).  visible(mws1gt_3,416). happensAt(walking(mws1gt_3),416). orientation(mws1gt_7,174,416). coords(mws1gt_7,70,192,416).  visible(mws1gt_7,416). happensAt(walking(mws1gt_7),416).
orientation(mws1gt_3,165,417). coords(mws1gt_3,33,160,417).  visible(mws1gt_3,417). happensAt(walking(mws1gt_3),417). orientation(mws1gt_7,171,417). coords(mws1gt_7,70,192,417).  visible(mws1gt_7,417). happensAt(walking(mws1gt_7),417).
orientation(mws1gt_3,165,418). coords(mws1gt_3,33,160,418).  visible(mws1gt_3,418). happensAt(walking(mws1gt_3),418). orientation(mws1gt_7,168,418). coords(mws1gt_7,71,192,418).  visible(mws1gt_7,418). happensAt(walking(mws1gt_7),418).
orientation(mws1gt_3,165,419). coords(mws1gt_3,33,160,419).  visible(mws1gt_3,419). happensAt(walking(mws1gt_3),419). orientation(mws1gt_7,169,419). coords(mws1gt_7,71,192,419).  visible(mws1gt_7,419). happensAt(walking(mws1gt_7),419).
orientation(mws1gt_3,165,420). coords(mws1gt_3,32,160,420).  visible(mws1gt_3,420). happensAt(walking(mws1gt_3),420). orientation(mws1gt_7,168,420). coords(mws1gt_7,72,192,420).  visible(mws1gt_7,420). happensAt(walking(mws1gt_7),420).
orientation(mws1gt_3,165,421). coords(mws1gt_3,32,160,421).  visible(mws1gt_3,421). happensAt(walking(mws1gt_3),421). orientation(mws1gt_7,168,421). coords(mws1gt_7,73,193,421).  visible(mws1gt_7,421). happensAt(walking(mws1gt_7),421).
orientation(mws1gt_3,165,422). coords(mws1gt_3,32,160,422).  visible(mws1gt_3,422). happensAt(walking(mws1gt_3),422). orientation(mws1gt_7,163,422). coords(mws1gt_7,73,193,422).  visible(mws1gt_7,422). happensAt(walking(mws1gt_7),422).
orientation(mws1gt_3,165,423). coords(mws1gt_3,32,160,423).  visible(mws1gt_3,423). happensAt(walking(mws1gt_3),423). orientation(mws1gt_7,163,423). coords(mws1gt_7,73,193,423).  visible(mws1gt_7,423). happensAt(walking(mws1gt_7),423).
orientation(mws1gt_3,165,424). coords(mws1gt_3,32,160,424).  visible(mws1gt_3,424). happensAt(walking(mws1gt_3),424). orientation(mws1gt_7,164,424). coords(mws1gt_7,73,193,424).  visible(mws1gt_7,424). happensAt(walking(mws1gt_7),424).
orientation(mws1gt_3,165,425). coords(mws1gt_3,31,160,425).  visible(mws1gt_3,425). happensAt(walking(mws1gt_3),425). orientation(mws1gt_7,164,425). coords(mws1gt_7,73,194,425).  visible(mws1gt_7,425). happensAt(walking(mws1gt_7),425).
orientation(mws1gt_3,165,426). coords(mws1gt_3,31,160,426).  visible(mws1gt_3,426). happensAt(walking(mws1gt_3),426). orientation(mws1gt_7,165,426). coords(mws1gt_7,73,194,426).  visible(mws1gt_7,426). happensAt(walking(mws1gt_7),426).
orientation(mws1gt_3,165,427). coords(mws1gt_3,31,160,427).  visible(mws1gt_3,427). happensAt(walking(mws1gt_3),427). orientation(mws1gt_7,166,427). coords(mws1gt_7,73,195,427).  visible(mws1gt_7,427). happensAt(walking(mws1gt_7),427).
orientation(mws1gt_3,165,428). coords(mws1gt_3,31,160,428).  visible(mws1gt_3,428). happensAt(walking(mws1gt_3),428). orientation(mws1gt_7,168,428). coords(mws1gt_7,73,195,428).  visible(mws1gt_7,428). happensAt(walking(mws1gt_7),428).
orientation(mws1gt_3,165,429). coords(mws1gt_3,31,160,429).  visible(mws1gt_3,429). happensAt(walking(mws1gt_3),429). orientation(mws1gt_7,166,429). coords(mws1gt_7,75,195,429).  visible(mws1gt_7,429). happensAt(walking(mws1gt_7),429).
orientation(mws1gt_3,165,430). coords(mws1gt_3,31,160,430).  visible(mws1gt_3,430). happensAt(walking(mws1gt_3),430). orientation(mws1gt_7,164,430). coords(mws1gt_7,76,195,430).  visible(mws1gt_7,430). happensAt(walking(mws1gt_7),430).
orientation(mws1gt_3,165,431). coords(mws1gt_3,30,160,431).  visible(mws1gt_3,431). happensAt(walking(mws1gt_3),431). orientation(mws1gt_7,165,431). coords(mws1gt_7,77,195,431).  visible(mws1gt_7,431). happensAt(walking(mws1gt_7),431).
orientation(mws1gt_3,165,432). coords(mws1gt_3,30,160,432).  visible(mws1gt_3,432). happensAt(walking(mws1gt_3),432). orientation(mws1gt_7,165,432). coords(mws1gt_7,77,195,432).  visible(mws1gt_7,432). happensAt(walking(mws1gt_7),432).
orientation(mws1gt_3,165,433). coords(mws1gt_3,30,160,433).  happensAt(disappear(mws1gt_3),433). happensAt(walking(mws1gt_3),433). orientation(mws1gt_7,161,433). coords(mws1gt_7,78,196,433).  visible(mws1gt_7,433). happensAt(walking(mws1gt_7),433).
#show holdsAt/2.