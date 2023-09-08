
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

	orientation(ms3ggt_0,158,410). coords(ms3ggt_0,131,140,410).  visible(ms3ggt_0,410). happensAt(walking(ms3ggt_0),410). orientation(ms3ggt_1,153,410). coords(ms3ggt_1,153,158,410).  visible(ms3ggt_1,410). happensAt(walking(ms3ggt_1),410).
orientation(ms3ggt_0,158,411). coords(ms3ggt_0,130,141,411).  visible(ms3ggt_0,411). happensAt(walking(ms3ggt_0),411). orientation(ms3ggt_1,153,411). coords(ms3ggt_1,154,158,411).  visible(ms3ggt_1,411). happensAt(walking(ms3ggt_1),411).
orientation(ms3ggt_0,158,412). coords(ms3ggt_0,130,141,412).  visible(ms3ggt_0,412). happensAt(walking(ms3ggt_0),412). orientation(ms3ggt_1,153,412). coords(ms3ggt_1,158,159,412).  visible(ms3ggt_1,412). happensAt(walking(ms3ggt_1),412).
orientation(ms3ggt_0,158,413). coords(ms3ggt_0,130,141,413).  visible(ms3ggt_0,413). happensAt(walking(ms3ggt_0),413). orientation(ms3ggt_1,153,413). coords(ms3ggt_1,159,159,413).  visible(ms3ggt_1,413). happensAt(walking(ms3ggt_1),413).
orientation(ms3ggt_0,162,414). coords(ms3ggt_0,128,142,414).  visible(ms3ggt_0,414). happensAt(walking(ms3ggt_0),414). orientation(ms3ggt_1,160,414). coords(ms3ggt_1,161,158,414).  visible(ms3ggt_1,414). happensAt(walking(ms3ggt_1),414).
orientation(ms3ggt_0,162,415). coords(ms3ggt_0,126,142,415).  visible(ms3ggt_0,415). happensAt(walking(ms3ggt_0),415). orientation(ms3ggt_1,160,415). coords(ms3ggt_1,163,158,415).  visible(ms3ggt_1,415). happensAt(walking(ms3ggt_1),415).
orientation(ms3ggt_0,162,416). coords(ms3ggt_0,125,142,416).  visible(ms3ggt_0,416). happensAt(walking(ms3ggt_0),416). orientation(ms3ggt_1,160,416). coords(ms3ggt_1,165,158,416).  visible(ms3ggt_1,416). happensAt(walking(ms3ggt_1),416).
orientation(ms3ggt_0,162,417). coords(ms3ggt_0,123,142,417).  visible(ms3ggt_0,417). happensAt(walking(ms3ggt_0),417). orientation(ms3ggt_1,160,417). coords(ms3ggt_1,167,158,417).  visible(ms3ggt_1,417). happensAt(walking(ms3ggt_1),417).
orientation(ms3ggt_0,164,418). coords(ms3ggt_0,122,143,418).  visible(ms3ggt_0,418). happensAt(walking(ms3ggt_0),418). orientation(ms3ggt_1,160,418). coords(ms3ggt_1,169,157,418).  visible(ms3ggt_1,418). happensAt(walking(ms3ggt_1),418).
orientation(ms3ggt_0,164,419). coords(ms3ggt_0,122,144,419).  visible(ms3ggt_0,419). happensAt(walking(ms3ggt_0),419). orientation(ms3ggt_1,163,419). coords(ms3ggt_1,170,157,419).  visible(ms3ggt_1,419). happensAt(walking(ms3ggt_1),419).
orientation(ms3ggt_0,164,420). coords(ms3ggt_0,120,144,420).  visible(ms3ggt_0,420). happensAt(walking(ms3ggt_0),420). orientation(ms3ggt_1,163,420). coords(ms3ggt_1,170,158,420).  visible(ms3ggt_1,420). happensAt(walking(ms3ggt_1),420).
orientation(ms3ggt_0,157,421). coords(ms3ggt_0,121,145,421).  visible(ms3ggt_0,421). happensAt(walking(ms3ggt_0),421). orientation(ms3ggt_1,163,421). coords(ms3ggt_1,171,158,421).  visible(ms3ggt_1,421). happensAt(walking(ms3ggt_1),421).
orientation(ms3ggt_0,157,422). coords(ms3ggt_0,121,145,422).  visible(ms3ggt_0,422). happensAt(walking(ms3ggt_0),422). orientation(ms3ggt_1,163,422). coords(ms3ggt_1,172,158,422).  visible(ms3ggt_1,422). happensAt(walking(ms3ggt_1),422).
orientation(ms3ggt_0,157,423). coords(ms3ggt_0,121,145,423).  visible(ms3ggt_0,423). happensAt(walking(ms3ggt_0),423). orientation(ms3ggt_1,163,423). coords(ms3ggt_1,172,157,423).  visible(ms3ggt_1,423). happensAt(walking(ms3ggt_1),423).
orientation(ms3ggt_0,157,424). coords(ms3ggt_0,120,145,424).  visible(ms3ggt_0,424). happensAt(walking(ms3ggt_0),424). orientation(ms3ggt_1,163,424). coords(ms3ggt_1,173,157,424).  visible(ms3ggt_1,424). happensAt(walking(ms3ggt_1),424).
orientation(ms3ggt_0,157,425). coords(ms3ggt_0,120,145,425).  visible(ms3ggt_0,425). happensAt(walking(ms3ggt_0),425). orientation(ms3ggt_1,163,425). coords(ms3ggt_1,173,157,425).  visible(ms3ggt_1,425). happensAt(walking(ms3ggt_1),425).
orientation(ms3ggt_0,157,426). coords(ms3ggt_0,119,145,426).  visible(ms3ggt_0,426). happensAt(walking(ms3ggt_0),426). orientation(ms3ggt_1,163,426). coords(ms3ggt_1,174,156,426).  visible(ms3ggt_1,426). happensAt(walking(ms3ggt_1),426).
orientation(ms3ggt_0,157,427). coords(ms3ggt_0,119,145,427).  visible(ms3ggt_0,427). happensAt(walking(ms3ggt_0),427). orientation(ms3ggt_1,163,427). coords(ms3ggt_1,174,156,427).  visible(ms3ggt_1,427). happensAt(walking(ms3ggt_1),427).
orientation(ms3ggt_0,157,428). coords(ms3ggt_0,118,145,428).  visible(ms3ggt_0,428). happensAt(walking(ms3ggt_0),428). orientation(ms3ggt_1,154,428). coords(ms3ggt_1,175,157,428).  visible(ms3ggt_1,428). happensAt(walking(ms3ggt_1),428).
orientation(ms3ggt_0,157,429). coords(ms3ggt_0,118,145,429).  visible(ms3ggt_0,429). happensAt(walking(ms3ggt_0),429). orientation(ms3ggt_1,154,429). coords(ms3ggt_1,175,157,429).  visible(ms3ggt_1,429). happensAt(walking(ms3ggt_1),429).
orientation(ms3ggt_0,157,430). coords(ms3ggt_0,116,145,430).  visible(ms3ggt_0,430). happensAt(walking(ms3ggt_0),430). orientation(ms3ggt_1,154,430). coords(ms3ggt_1,176,157,430).  visible(ms3ggt_1,430). happensAt(walking(ms3ggt_1),430).
orientation(ms3ggt_0,160,431). coords(ms3ggt_0,114,144,431).  visible(ms3ggt_0,431). happensAt(walking(ms3ggt_0),431). orientation(ms3ggt_1,158,431). coords(ms3ggt_1,181,159,431).  visible(ms3ggt_1,431). happensAt(walking(ms3ggt_1),431).
orientation(ms3ggt_0,160,432). coords(ms3ggt_0,112,144,432).  visible(ms3ggt_0,432). happensAt(walking(ms3ggt_0),432). orientation(ms3ggt_1,158,432). coords(ms3ggt_1,184,159,432).  visible(ms3ggt_1,432). happensAt(walking(ms3ggt_1),432).
orientation(ms3ggt_0,160,433). coords(ms3ggt_0,111,144,433).  visible(ms3ggt_0,433). happensAt(walking(ms3ggt_0),433). orientation(ms3ggt_1,158,433). coords(ms3ggt_1,186,161,433).  visible(ms3ggt_1,433). happensAt(walking(ms3ggt_1),433).
orientation(ms3ggt_0,160,434). coords(ms3ggt_0,109,144,434).  visible(ms3ggt_0,434). happensAt(walking(ms3ggt_0),434). orientation(ms3ggt_1,158,434). coords(ms3ggt_1,188,161,434).  visible(ms3ggt_1,434). happensAt(walking(ms3ggt_1),434).
#show holdsAt/2.