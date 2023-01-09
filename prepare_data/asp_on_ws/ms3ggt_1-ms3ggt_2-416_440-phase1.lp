
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

	orientation(ms3ggt_1,160,416). coords(ms3ggt_1,165,158,416).  visible(ms3ggt_1,416). happensAt(walking(ms3ggt_1),416). orientation(ms3ggt_2,162,416). coords(ms3ggt_2,125,178,416).  visible(ms3ggt_2,416). happensAt(walking(ms3ggt_2),416).
orientation(ms3ggt_1,160,417). coords(ms3ggt_1,167,158,417).  visible(ms3ggt_1,417). happensAt(walking(ms3ggt_1),417). orientation(ms3ggt_2,162,417). coords(ms3ggt_2,125,177,417).  visible(ms3ggt_2,417). happensAt(walking(ms3ggt_2),417).
orientation(ms3ggt_1,160,418). coords(ms3ggt_1,169,157,418).  visible(ms3ggt_1,418). happensAt(walking(ms3ggt_1),418). orientation(ms3ggt_2,162,418). coords(ms3ggt_2,125,176,418).  visible(ms3ggt_2,418). happensAt(walking(ms3ggt_2),418).
orientation(ms3ggt_1,163,419). coords(ms3ggt_1,170,157,419).  visible(ms3ggt_1,419). happensAt(walking(ms3ggt_1),419). orientation(ms3ggt_2,166,419). coords(ms3ggt_2,125,176,419).  visible(ms3ggt_2,419). happensAt(walking(ms3ggt_2),419).
orientation(ms3ggt_1,163,420). coords(ms3ggt_1,170,158,420).  visible(ms3ggt_1,420). happensAt(walking(ms3ggt_1),420). orientation(ms3ggt_2,166,420). coords(ms3ggt_2,124,176,420).  visible(ms3ggt_2,420). happensAt(walking(ms3ggt_2),420).
orientation(ms3ggt_1,163,421). coords(ms3ggt_1,171,158,421).  visible(ms3ggt_1,421). happensAt(walking(ms3ggt_1),421). orientation(ms3ggt_2,166,421). coords(ms3ggt_2,124,176,421).  visible(ms3ggt_2,421). happensAt(walking(ms3ggt_2),421).
orientation(ms3ggt_1,163,422). coords(ms3ggt_1,172,158,422).  visible(ms3ggt_1,422). happensAt(walking(ms3ggt_1),422). orientation(ms3ggt_2,166,422). coords(ms3ggt_2,122,176,422).  visible(ms3ggt_2,422). happensAt(walking(ms3ggt_2),422).
orientation(ms3ggt_1,163,423). coords(ms3ggt_1,172,157,423).  visible(ms3ggt_1,423). happensAt(walking(ms3ggt_1),423). orientation(ms3ggt_2,166,423). coords(ms3ggt_2,121,176,423).  visible(ms3ggt_2,423). happensAt(walking(ms3ggt_2),423).
orientation(ms3ggt_1,163,424). coords(ms3ggt_1,173,157,424).  visible(ms3ggt_1,424). happensAt(walking(ms3ggt_1),424). orientation(ms3ggt_2,166,424). coords(ms3ggt_2,120,177,424).  visible(ms3ggt_2,424). happensAt(walking(ms3ggt_2),424).
orientation(ms3ggt_1,163,425). coords(ms3ggt_1,173,157,425).  visible(ms3ggt_1,425). happensAt(walking(ms3ggt_1),425). orientation(ms3ggt_2,165,425). coords(ms3ggt_2,120,177,425).  visible(ms3ggt_2,425). happensAt(walking(ms3ggt_2),425).
orientation(ms3ggt_1,163,426). coords(ms3ggt_1,174,156,426).  visible(ms3ggt_1,426). happensAt(walking(ms3ggt_1),426). orientation(ms3ggt_2,159,426). coords(ms3ggt_2,118,177,426).  visible(ms3ggt_2,426). happensAt(walking(ms3ggt_2),426).
orientation(ms3ggt_1,163,427). coords(ms3ggt_1,174,156,427).  visible(ms3ggt_1,427). happensAt(walking(ms3ggt_1),427). orientation(ms3ggt_2,159,427). coords(ms3ggt_2,117,177,427).  visible(ms3ggt_2,427). happensAt(walking(ms3ggt_2),427).
orientation(ms3ggt_1,154,428). coords(ms3ggt_1,175,157,428).  visible(ms3ggt_1,428). happensAt(walking(ms3ggt_1),428). orientation(ms3ggt_2,159,428). coords(ms3ggt_2,117,177,428).  visible(ms3ggt_2,428). happensAt(walking(ms3ggt_2),428).
orientation(ms3ggt_1,154,429). coords(ms3ggt_1,175,157,429).  visible(ms3ggt_1,429). happensAt(walking(ms3ggt_1),429). orientation(ms3ggt_2,159,429). coords(ms3ggt_2,117,176,429).  visible(ms3ggt_2,429). happensAt(walking(ms3ggt_2),429).
orientation(ms3ggt_1,154,430). coords(ms3ggt_1,176,157,430).  visible(ms3ggt_1,430). happensAt(walking(ms3ggt_1),430). orientation(ms3ggt_2,155,430). coords(ms3ggt_2,116,176,430).  visible(ms3ggt_2,430). happensAt(walking(ms3ggt_2),430).
orientation(ms3ggt_1,158,431). coords(ms3ggt_1,181,159,431).  visible(ms3ggt_1,431). happensAt(walking(ms3ggt_1),431). orientation(ms3ggt_2,155,431). coords(ms3ggt_2,116,175,431).  visible(ms3ggt_2,431). happensAt(walking(ms3ggt_2),431).
orientation(ms3ggt_1,158,432). coords(ms3ggt_1,184,159,432).  visible(ms3ggt_1,432). happensAt(walking(ms3ggt_1),432). orientation(ms3ggt_2,155,432). coords(ms3ggt_2,116,174,432).  visible(ms3ggt_2,432). happensAt(walking(ms3ggt_2),432).
orientation(ms3ggt_1,158,433). coords(ms3ggt_1,186,161,433).  visible(ms3ggt_1,433). happensAt(walking(ms3ggt_1),433). orientation(ms3ggt_2,155,433). coords(ms3ggt_2,115,174,433).  visible(ms3ggt_2,433). happensAt(walking(ms3ggt_2),433).
orientation(ms3ggt_1,158,434). coords(ms3ggt_1,188,161,434).  visible(ms3ggt_1,434). happensAt(walking(ms3ggt_1),434). orientation(ms3ggt_2,155,434). coords(ms3ggt_2,115,173,434).  visible(ms3ggt_2,434). happensAt(walking(ms3ggt_2),434).
orientation(ms3ggt_1,158,435). coords(ms3ggt_1,188,161,435).  visible(ms3ggt_1,435). happensAt(walking(ms3ggt_1),435). orientation(ms3ggt_2,155,435). coords(ms3ggt_2,115,173,435).  visible(ms3ggt_2,435). happensAt(walking(ms3ggt_2),435).
orientation(ms3ggt_1,158,436). coords(ms3ggt_1,189,161,436).  visible(ms3ggt_1,436). happensAt(walking(ms3ggt_1),436). orientation(ms3ggt_2,155,436). coords(ms3ggt_2,114,171,436).  visible(ms3ggt_2,436). happensAt(walking(ms3ggt_2),436).
orientation(ms3ggt_1,158,437). coords(ms3ggt_1,189,161,437).  visible(ms3ggt_1,437). happensAt(walking(ms3ggt_1),437). orientation(ms3ggt_2,155,437). coords(ms3ggt_2,112,171,437).  visible(ms3ggt_2,437). happensAt(walking(ms3ggt_2),437).
orientation(ms3ggt_1,158,438). coords(ms3ggt_1,191,161,438).  visible(ms3ggt_1,438). happensAt(walking(ms3ggt_1),438). orientation(ms3ggt_2,155,438). coords(ms3ggt_2,111,170,438).  visible(ms3ggt_2,438). happensAt(walking(ms3ggt_2),438).
orientation(ms3ggt_1,158,439). coords(ms3ggt_1,191,161,439).  visible(ms3ggt_1,439). happensAt(walking(ms3ggt_1),439). orientation(ms3ggt_2,155,439). coords(ms3ggt_2,111,169,439).  visible(ms3ggt_2,439). happensAt(walking(ms3ggt_2),439).
orientation(ms3ggt_1,158,440). coords(ms3ggt_1,192,161,440).  visible(ms3ggt_1,440). happensAt(walking(ms3ggt_1),440). orientation(ms3ggt_2,155,440). coords(ms3ggt_2,110,169,440).  visible(ms3ggt_2,440). happensAt(walking(ms3ggt_2),440).
#show holdsAt/2.