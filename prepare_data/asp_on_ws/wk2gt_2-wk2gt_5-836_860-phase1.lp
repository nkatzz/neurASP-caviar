
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

	orientation(wk2gt_2,142,836). coords(wk2gt_2,99,63,836).  visible(wk2gt_2,836). happensAt(inactive(wk2gt_2),836). orientation(wk2gt_5,156,836). coords(wk2gt_5,37,110,836).  visible(wk2gt_5,836). happensAt(walking(wk2gt_5),836).
orientation(wk2gt_2,142,837). coords(wk2gt_2,99,63,837).  visible(wk2gt_2,837). happensAt(inactive(wk2gt_2),837). orientation(wk2gt_5,156,837). coords(wk2gt_5,37,110,837).  visible(wk2gt_5,837). happensAt(walking(wk2gt_5),837).
orientation(wk2gt_2,142,838). coords(wk2gt_2,99,63,838).  visible(wk2gt_2,838). happensAt(inactive(wk2gt_2),838). orientation(wk2gt_5,156,838). coords(wk2gt_5,37,109,838).  visible(wk2gt_5,838). happensAt(walking(wk2gt_5),838).
orientation(wk2gt_2,142,839). coords(wk2gt_2,99,63,839).  visible(wk2gt_2,839). happensAt(inactive(wk2gt_2),839). orientation(wk2gt_5,156,839). coords(wk2gt_5,37,109,839).  visible(wk2gt_5,839). happensAt(walking(wk2gt_5),839).
orientation(wk2gt_2,142,840). coords(wk2gt_2,99,63,840).  visible(wk2gt_2,840). happensAt(inactive(wk2gt_2),840). orientation(wk2gt_5,156,840). coords(wk2gt_5,38,109,840).  visible(wk2gt_5,840). happensAt(walking(wk2gt_5),840).
orientation(wk2gt_2,142,841). coords(wk2gt_2,99,63,841).  visible(wk2gt_2,841). happensAt(inactive(wk2gt_2),841). orientation(wk2gt_5,156,841). coords(wk2gt_5,40,108,841).  visible(wk2gt_5,841). happensAt(walking(wk2gt_5),841).
orientation(wk2gt_2,142,842). coords(wk2gt_2,99,63,842).  visible(wk2gt_2,842). happensAt(inactive(wk2gt_2),842). orientation(wk2gt_5,156,842). coords(wk2gt_5,40,108,842).  visible(wk2gt_5,842). happensAt(walking(wk2gt_5),842).
orientation(wk2gt_2,142,843). coords(wk2gt_2,99,63,843).  visible(wk2gt_2,843). happensAt(inactive(wk2gt_2),843). orientation(wk2gt_5,156,843). coords(wk2gt_5,40,108,843).  visible(wk2gt_5,843). happensAt(walking(wk2gt_5),843).
orientation(wk2gt_2,142,844). coords(wk2gt_2,99,63,844).  visible(wk2gt_2,844). happensAt(inactive(wk2gt_2),844). orientation(wk2gt_5,156,844). coords(wk2gt_5,40,108,844).  visible(wk2gt_5,844). happensAt(walking(wk2gt_5),844).
orientation(wk2gt_2,142,845). coords(wk2gt_2,99,63,845).  visible(wk2gt_2,845). happensAt(inactive(wk2gt_2),845). orientation(wk2gt_5,156,845). coords(wk2gt_5,41,107,845).  visible(wk2gt_5,845). happensAt(walking(wk2gt_5),845).
orientation(wk2gt_2,142,846). coords(wk2gt_2,99,63,846).  visible(wk2gt_2,846). happensAt(inactive(wk2gt_2),846). orientation(wk2gt_5,156,846). coords(wk2gt_5,41,107,846).  visible(wk2gt_5,846). happensAt(walking(wk2gt_5),846).
orientation(wk2gt_2,142,847). coords(wk2gt_2,99,63,847).  visible(wk2gt_2,847). happensAt(inactive(wk2gt_2),847). orientation(wk2gt_5,156,847). coords(wk2gt_5,41,106,847).  visible(wk2gt_5,847). happensAt(walking(wk2gt_5),847).
orientation(wk2gt_2,142,848). coords(wk2gt_2,99,63,848).  visible(wk2gt_2,848). happensAt(inactive(wk2gt_2),848). orientation(wk2gt_5,156,848). coords(wk2gt_5,41,106,848).  visible(wk2gt_5,848). happensAt(walking(wk2gt_5),848).
orientation(wk2gt_2,142,849). coords(wk2gt_2,99,63,849).  visible(wk2gt_2,849). happensAt(inactive(wk2gt_2),849). orientation(wk2gt_5,156,849). coords(wk2gt_5,41,106,849).  visible(wk2gt_5,849). happensAt(walking(wk2gt_5),849).
orientation(wk2gt_2,142,850). coords(wk2gt_2,99,63,850).  visible(wk2gt_2,850). happensAt(inactive(wk2gt_2),850). orientation(wk2gt_5,156,850). coords(wk2gt_5,41,106,850).  visible(wk2gt_5,850). happensAt(walking(wk2gt_5),850).
orientation(wk2gt_2,142,851). coords(wk2gt_2,99,63,851).  visible(wk2gt_2,851). happensAt(inactive(wk2gt_2),851). orientation(wk2gt_5,156,851). coords(wk2gt_5,41,106,851).  visible(wk2gt_5,851). happensAt(walking(wk2gt_5),851).
orientation(wk2gt_2,142,852). coords(wk2gt_2,99,63,852).  visible(wk2gt_2,852). happensAt(inactive(wk2gt_2),852). orientation(wk2gt_5,156,852). coords(wk2gt_5,42,105,852).  visible(wk2gt_5,852). happensAt(walking(wk2gt_5),852).
orientation(wk2gt_2,142,853). coords(wk2gt_2,99,63,853).  visible(wk2gt_2,853). happensAt(inactive(wk2gt_2),853). orientation(wk2gt_5,156,853). coords(wk2gt_5,43,105,853).  visible(wk2gt_5,853). happensAt(walking(wk2gt_5),853).
orientation(wk2gt_2,142,854). coords(wk2gt_2,99,63,854).  visible(wk2gt_2,854). happensAt(inactive(wk2gt_2),854). orientation(wk2gt_5,156,854). coords(wk2gt_5,43,105,854).  visible(wk2gt_5,854). happensAt(walking(wk2gt_5),854).
orientation(wk2gt_2,142,855). coords(wk2gt_2,99,63,855).  visible(wk2gt_2,855). happensAt(inactive(wk2gt_2),855). orientation(wk2gt_5,156,855). coords(wk2gt_5,43,105,855).  visible(wk2gt_5,855). happensAt(walking(wk2gt_5),855).
orientation(wk2gt_2,142,856). coords(wk2gt_2,99,63,856).  visible(wk2gt_2,856). happensAt(inactive(wk2gt_2),856). orientation(wk2gt_5,156,856). coords(wk2gt_5,43,105,856).  visible(wk2gt_5,856). happensAt(walking(wk2gt_5),856).
orientation(wk2gt_2,142,857). coords(wk2gt_2,99,63,857).  visible(wk2gt_2,857). happensAt(inactive(wk2gt_2),857). orientation(wk2gt_5,156,857). coords(wk2gt_5,43,105,857).  visible(wk2gt_5,857). happensAt(walking(wk2gt_5),857).
orientation(wk2gt_2,142,858). coords(wk2gt_2,99,63,858).  visible(wk2gt_2,858). happensAt(inactive(wk2gt_2),858). orientation(wk2gt_5,156,858). coords(wk2gt_5,44,105,858).  visible(wk2gt_5,858). happensAt(walking(wk2gt_5),858).
orientation(wk2gt_2,142,859). coords(wk2gt_2,99,63,859).  visible(wk2gt_2,859). happensAt(inactive(wk2gt_2),859). orientation(wk2gt_5,156,859). coords(wk2gt_5,44,105,859).  visible(wk2gt_5,859). happensAt(walking(wk2gt_5),859).
orientation(wk2gt_2,142,860). coords(wk2gt_2,99,63,860).  visible(wk2gt_2,860). happensAt(inactive(wk2gt_2),860). orientation(wk2gt_5,156,860). coords(wk2gt_5,44,104,860).  visible(wk2gt_5,860). happensAt(walking(wk2gt_5),860).
#show holdsAt/2.