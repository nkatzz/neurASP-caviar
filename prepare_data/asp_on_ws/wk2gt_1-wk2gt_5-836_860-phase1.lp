
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

	orientation(wk2gt_1,125,836). coords(wk2gt_1,115,53,836).  visible(wk2gt_1,836). happensAt(inactive(wk2gt_1),836). orientation(wk2gt_5,156,836). coords(wk2gt_5,37,110,836).  visible(wk2gt_5,836). happensAt(walking(wk2gt_5),836).
orientation(wk2gt_1,125,837). coords(wk2gt_1,115,53,837).  visible(wk2gt_1,837). happensAt(inactive(wk2gt_1),837). orientation(wk2gt_5,156,837). coords(wk2gt_5,37,110,837).  visible(wk2gt_5,837). happensAt(walking(wk2gt_5),837).
orientation(wk2gt_1,125,838). coords(wk2gt_1,115,53,838).  visible(wk2gt_1,838). happensAt(inactive(wk2gt_1),838). orientation(wk2gt_5,156,838). coords(wk2gt_5,37,109,838).  visible(wk2gt_5,838). happensAt(walking(wk2gt_5),838).
orientation(wk2gt_1,125,839). coords(wk2gt_1,115,53,839).  visible(wk2gt_1,839). happensAt(inactive(wk2gt_1),839). orientation(wk2gt_5,156,839). coords(wk2gt_5,37,109,839).  visible(wk2gt_5,839). happensAt(walking(wk2gt_5),839).
orientation(wk2gt_1,125,840). coords(wk2gt_1,115,53,840).  visible(wk2gt_1,840). happensAt(inactive(wk2gt_1),840). orientation(wk2gt_5,156,840). coords(wk2gt_5,38,109,840).  visible(wk2gt_5,840). happensAt(walking(wk2gt_5),840).
orientation(wk2gt_1,125,841). coords(wk2gt_1,115,53,841).  visible(wk2gt_1,841). happensAt(inactive(wk2gt_1),841). orientation(wk2gt_5,156,841). coords(wk2gt_5,40,108,841).  visible(wk2gt_5,841). happensAt(walking(wk2gt_5),841).
orientation(wk2gt_1,125,842). coords(wk2gt_1,115,53,842).  visible(wk2gt_1,842). happensAt(inactive(wk2gt_1),842). orientation(wk2gt_5,156,842). coords(wk2gt_5,40,108,842).  visible(wk2gt_5,842). happensAt(walking(wk2gt_5),842).
orientation(wk2gt_1,125,843). coords(wk2gt_1,115,53,843).  visible(wk2gt_1,843). happensAt(inactive(wk2gt_1),843). orientation(wk2gt_5,156,843). coords(wk2gt_5,40,108,843).  visible(wk2gt_5,843). happensAt(walking(wk2gt_5),843).
orientation(wk2gt_1,125,844). coords(wk2gt_1,115,53,844).  visible(wk2gt_1,844). happensAt(inactive(wk2gt_1),844). orientation(wk2gt_5,156,844). coords(wk2gt_5,40,108,844).  visible(wk2gt_5,844). happensAt(walking(wk2gt_5),844).
orientation(wk2gt_1,125,845). coords(wk2gt_1,115,53,845).  visible(wk2gt_1,845). happensAt(inactive(wk2gt_1),845). orientation(wk2gt_5,156,845). coords(wk2gt_5,41,107,845).  visible(wk2gt_5,845). happensAt(walking(wk2gt_5),845).
orientation(wk2gt_1,125,846). coords(wk2gt_1,115,53,846).  visible(wk2gt_1,846). happensAt(inactive(wk2gt_1),846). orientation(wk2gt_5,156,846). coords(wk2gt_5,41,107,846).  visible(wk2gt_5,846). happensAt(walking(wk2gt_5),846).
orientation(wk2gt_1,127,847). coords(wk2gt_1,115,53,847).  visible(wk2gt_1,847). happensAt(inactive(wk2gt_1),847). orientation(wk2gt_5,156,847). coords(wk2gt_5,41,106,847).  visible(wk2gt_5,847). happensAt(walking(wk2gt_5),847).
orientation(wk2gt_1,127,848). coords(wk2gt_1,115,53,848).  visible(wk2gt_1,848). happensAt(inactive(wk2gt_1),848). orientation(wk2gt_5,156,848). coords(wk2gt_5,41,106,848).  visible(wk2gt_5,848). happensAt(walking(wk2gt_5),848).
orientation(wk2gt_1,127,849). coords(wk2gt_1,115,53,849).  visible(wk2gt_1,849). happensAt(inactive(wk2gt_1),849). orientation(wk2gt_5,156,849). coords(wk2gt_5,41,106,849).  visible(wk2gt_5,849). happensAt(walking(wk2gt_5),849).
orientation(wk2gt_1,127,850). coords(wk2gt_1,115,53,850).  visible(wk2gt_1,850). happensAt(inactive(wk2gt_1),850). orientation(wk2gt_5,156,850). coords(wk2gt_5,41,106,850).  visible(wk2gt_5,850). happensAt(walking(wk2gt_5),850).
orientation(wk2gt_1,127,851). coords(wk2gt_1,115,53,851).  visible(wk2gt_1,851). happensAt(inactive(wk2gt_1),851). orientation(wk2gt_5,156,851). coords(wk2gt_5,41,106,851).  visible(wk2gt_5,851). happensAt(walking(wk2gt_5),851).
orientation(wk2gt_1,127,852). coords(wk2gt_1,115,53,852).  visible(wk2gt_1,852). happensAt(inactive(wk2gt_1),852). orientation(wk2gt_5,156,852). coords(wk2gt_5,42,105,852).  visible(wk2gt_5,852). happensAt(walking(wk2gt_5),852).
orientation(wk2gt_1,127,853). coords(wk2gt_1,115,53,853).  visible(wk2gt_1,853). happensAt(inactive(wk2gt_1),853). orientation(wk2gt_5,156,853). coords(wk2gt_5,43,105,853).  visible(wk2gt_5,853). happensAt(walking(wk2gt_5),853).
orientation(wk2gt_1,127,854). coords(wk2gt_1,115,53,854).  visible(wk2gt_1,854). happensAt(inactive(wk2gt_1),854). orientation(wk2gt_5,156,854). coords(wk2gt_5,43,105,854).  visible(wk2gt_5,854). happensAt(walking(wk2gt_5),854).
orientation(wk2gt_1,127,855). coords(wk2gt_1,115,53,855).  visible(wk2gt_1,855). happensAt(inactive(wk2gt_1),855). orientation(wk2gt_5,156,855). coords(wk2gt_5,43,105,855).  visible(wk2gt_5,855). happensAt(walking(wk2gt_5),855).
orientation(wk2gt_1,127,856). coords(wk2gt_1,115,53,856).  visible(wk2gt_1,856). happensAt(inactive(wk2gt_1),856). orientation(wk2gt_5,156,856). coords(wk2gt_5,43,105,856).  visible(wk2gt_5,856). happensAt(walking(wk2gt_5),856).
orientation(wk2gt_1,127,857). coords(wk2gt_1,115,53,857).  visible(wk2gt_1,857). happensAt(inactive(wk2gt_1),857). orientation(wk2gt_5,156,857). coords(wk2gt_5,43,105,857).  visible(wk2gt_5,857). happensAt(walking(wk2gt_5),857).
orientation(wk2gt_1,127,858). coords(wk2gt_1,115,53,858).  visible(wk2gt_1,858). happensAt(inactive(wk2gt_1),858). orientation(wk2gt_5,156,858). coords(wk2gt_5,44,105,858).  visible(wk2gt_5,858). happensAt(walking(wk2gt_5),858).
orientation(wk2gt_1,127,859). coords(wk2gt_1,115,53,859).  visible(wk2gt_1,859). happensAt(inactive(wk2gt_1),859). orientation(wk2gt_5,156,859). coords(wk2gt_5,44,105,859).  visible(wk2gt_5,859). happensAt(walking(wk2gt_5),859).
orientation(wk2gt_1,127,860). coords(wk2gt_1,115,53,860).  visible(wk2gt_1,860). happensAt(inactive(wk2gt_1),860). orientation(wk2gt_5,156,860). coords(wk2gt_5,44,104,860).  visible(wk2gt_5,860). happensAt(walking(wk2gt_5),860).
#show holdsAt/2.