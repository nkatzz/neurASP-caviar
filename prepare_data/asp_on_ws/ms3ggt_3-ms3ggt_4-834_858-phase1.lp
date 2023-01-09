
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

	orientation(ms3ggt_3,156,834). coords(ms3ggt_3,37,146,834).  visible(ms3ggt_3,834). happensAt(walking(ms3ggt_3),834). orientation(ms3ggt_4,136,834). coords(ms3ggt_4,135,34,834).  visible(ms3ggt_4,834). happensAt(walking(ms3ggt_4),834).
orientation(ms3ggt_3,156,835). coords(ms3ggt_3,36,147,835).  visible(ms3ggt_3,835). happensAt(walking(ms3ggt_3),835). orientation(ms3ggt_4,136,835). coords(ms3ggt_4,135,34,835).  visible(ms3ggt_4,835). happensAt(walking(ms3ggt_4),835).
orientation(ms3ggt_3,156,836). coords(ms3ggt_3,37,149,836).  visible(ms3ggt_3,836). happensAt(walking(ms3ggt_3),836). orientation(ms3ggt_4,136,836). coords(ms3ggt_4,135,34,836).  visible(ms3ggt_4,836). happensAt(walking(ms3ggt_4),836).
orientation(ms3ggt_3,153,837). coords(ms3ggt_3,37,151,837).  visible(ms3ggt_3,837). happensAt(walking(ms3ggt_3),837). orientation(ms3ggt_4,136,837). coords(ms3ggt_4,135,35,837).  visible(ms3ggt_4,837). happensAt(walking(ms3ggt_4),837).
orientation(ms3ggt_3,153,838). coords(ms3ggt_3,37,153,838).  visible(ms3ggt_3,838). happensAt(walking(ms3ggt_3),838). orientation(ms3ggt_4,136,838). coords(ms3ggt_4,135,36,838).  visible(ms3ggt_4,838). happensAt(walking(ms3ggt_4),838).
orientation(ms3ggt_3,153,839). coords(ms3ggt_3,37,154,839).  visible(ms3ggt_3,839). happensAt(walking(ms3ggt_3),839). orientation(ms3ggt_4,136,839). coords(ms3ggt_4,135,36,839).  visible(ms3ggt_4,839). happensAt(walking(ms3ggt_4),839).
orientation(ms3ggt_3,153,840). coords(ms3ggt_3,37,154,840).  visible(ms3ggt_3,840). happensAt(walking(ms3ggt_3),840). orientation(ms3ggt_4,136,840). coords(ms3ggt_4,135,36,840).  visible(ms3ggt_4,840). happensAt(walking(ms3ggt_4),840).
orientation(ms3ggt_3,153,841). coords(ms3ggt_3,37,155,841).  visible(ms3ggt_3,841). happensAt(walking(ms3ggt_3),841). orientation(ms3ggt_4,136,841). coords(ms3ggt_4,135,36,841).  visible(ms3ggt_4,841). happensAt(walking(ms3ggt_4),841).
orientation(ms3ggt_3,153,842). coords(ms3ggt_3,37,156,842).  visible(ms3ggt_3,842). happensAt(walking(ms3ggt_3),842). orientation(ms3ggt_4,136,842). coords(ms3ggt_4,135,36,842).  visible(ms3ggt_4,842). happensAt(walking(ms3ggt_4),842).
orientation(ms3ggt_3,153,843). coords(ms3ggt_3,37,157,843).  visible(ms3ggt_3,843). happensAt(walking(ms3ggt_3),843). orientation(ms3ggt_4,136,843). coords(ms3ggt_4,135,37,843).  visible(ms3ggt_4,843). happensAt(walking(ms3ggt_4),843).
orientation(ms3ggt_3,153,844). coords(ms3ggt_3,37,158,844).  visible(ms3ggt_3,844). happensAt(walking(ms3ggt_3),844). orientation(ms3ggt_4,136,844). coords(ms3ggt_4,135,37,844).  visible(ms3ggt_4,844). happensAt(walking(ms3ggt_4),844).
orientation(ms3ggt_3,153,845). coords(ms3ggt_3,36,159,845).  visible(ms3ggt_3,845). happensAt(walking(ms3ggt_3),845). orientation(ms3ggt_4,140,845). coords(ms3ggt_4,135,38,845).  visible(ms3ggt_4,845). happensAt(walking(ms3ggt_4),845).
orientation(ms3ggt_3,153,846). coords(ms3ggt_3,35,160,846).  visible(ms3ggt_3,846). happensAt(walking(ms3ggt_3),846). orientation(ms3ggt_4,140,846). coords(ms3ggt_4,135,38,846).  visible(ms3ggt_4,846). happensAt(walking(ms3ggt_4),846).
orientation(ms3ggt_3,153,847). coords(ms3ggt_3,35,160,847).  visible(ms3ggt_3,847). happensAt(walking(ms3ggt_3),847). orientation(ms3ggt_4,140,847). coords(ms3ggt_4,135,38,847).  visible(ms3ggt_4,847). happensAt(walking(ms3ggt_4),847).
orientation(ms3ggt_3,153,848). coords(ms3ggt_3,35,161,848).  visible(ms3ggt_3,848). happensAt(walking(ms3ggt_3),848). orientation(ms3ggt_4,140,848). coords(ms3ggt_4,135,38,848).  visible(ms3ggt_4,848). happensAt(walking(ms3ggt_4),848).
orientation(ms3ggt_3,153,849). coords(ms3ggt_3,35,163,849).  visible(ms3ggt_3,849). happensAt(walking(ms3ggt_3),849). orientation(ms3ggt_4,140,849). coords(ms3ggt_4,135,38,849).  visible(ms3ggt_4,849). happensAt(walking(ms3ggt_4),849).
orientation(ms3ggt_3,162,850). coords(ms3ggt_3,35,163,850).  visible(ms3ggt_3,850). happensAt(walking(ms3ggt_3),850). orientation(ms3ggt_4,140,850). coords(ms3ggt_4,135,39,850).  visible(ms3ggt_4,850). happensAt(walking(ms3ggt_4),850).
orientation(ms3ggt_3,162,851). coords(ms3ggt_3,35,166,851).  visible(ms3ggt_3,851). happensAt(walking(ms3ggt_3),851). orientation(ms3ggt_4,140,851). coords(ms3ggt_4,135,39,851).  visible(ms3ggt_4,851). happensAt(walking(ms3ggt_4),851).
orientation(ms3ggt_3,162,852). coords(ms3ggt_3,35,166,852).  visible(ms3ggt_3,852). happensAt(walking(ms3ggt_3),852). orientation(ms3ggt_4,140,852). coords(ms3ggt_4,135,39,852).  visible(ms3ggt_4,852). happensAt(walking(ms3ggt_4),852).
orientation(ms3ggt_3,162,853). coords(ms3ggt_3,35,167,853).  visible(ms3ggt_3,853). happensAt(walking(ms3ggt_3),853). orientation(ms3ggt_4,140,853). coords(ms3ggt_4,135,39,853).  visible(ms3ggt_4,853). happensAt(walking(ms3ggt_4),853).
orientation(ms3ggt_3,162,854). coords(ms3ggt_3,34,170,854).  visible(ms3ggt_3,854). happensAt(walking(ms3ggt_3),854). orientation(ms3ggt_4,140,854). coords(ms3ggt_4,134,39,854).  visible(ms3ggt_4,854). happensAt(walking(ms3ggt_4),854).
orientation(ms3ggt_3,162,855). coords(ms3ggt_3,34,171,855).  visible(ms3ggt_3,855). happensAt(walking(ms3ggt_3),855). orientation(ms3ggt_4,140,855). coords(ms3ggt_4,134,39,855).  visible(ms3ggt_4,855). happensAt(walking(ms3ggt_4),855).
orientation(ms3ggt_3,162,856). coords(ms3ggt_3,34,171,856).  visible(ms3ggt_3,856). happensAt(walking(ms3ggt_3),856). orientation(ms3ggt_4,140,856). coords(ms3ggt_4,134,39,856).  visible(ms3ggt_4,856). happensAt(walking(ms3ggt_4),856).
orientation(ms3ggt_3,162,857). coords(ms3ggt_3,34,172,857).  visible(ms3ggt_3,857). happensAt(walking(ms3ggt_3),857). orientation(ms3ggt_4,140,857). coords(ms3ggt_4,134,39,857).  visible(ms3ggt_4,857). happensAt(walking(ms3ggt_4),857).
orientation(ms3ggt_3,162,858). coords(ms3ggt_3,33,172,858).  visible(ms3ggt_3,858). happensAt(walking(ms3ggt_3),858). orientation(ms3ggt_4,140,858). coords(ms3ggt_4,133,40,858).  visible(ms3ggt_4,858). happensAt(walking(ms3ggt_4),858).
#show holdsAt/2.