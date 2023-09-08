
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

	orientation(ms3ggt_3,153,822). coords(ms3ggt_3,41,133,822).  visible(ms3ggt_3,822). happensAt(walking(ms3ggt_3),822). orientation(ms3ggt_4,131,822). coords(ms3ggt_4,136,33,822).  visible(ms3ggt_4,822). happensAt(walking(ms3ggt_4),822).
orientation(ms3ggt_3,153,823). coords(ms3ggt_3,41,133,823).  visible(ms3ggt_3,823). happensAt(walking(ms3ggt_3),823). orientation(ms3ggt_4,131,823). coords(ms3ggt_4,136,33,823).  visible(ms3ggt_4,823). happensAt(walking(ms3ggt_4),823).
orientation(ms3ggt_3,150,824). coords(ms3ggt_3,40,137,824).  visible(ms3ggt_3,824). happensAt(walking(ms3ggt_3),824). orientation(ms3ggt_4,131,824). coords(ms3ggt_4,136,33,824).  visible(ms3ggt_4,824). happensAt(walking(ms3ggt_4),824).
orientation(ms3ggt_3,150,825). coords(ms3ggt_3,40,138,825).  visible(ms3ggt_3,825). happensAt(walking(ms3ggt_3),825). orientation(ms3ggt_4,131,825). coords(ms3ggt_4,136,33,825).  visible(ms3ggt_4,825). happensAt(walking(ms3ggt_4),825).
orientation(ms3ggt_3,150,826). coords(ms3ggt_3,40,138,826).  visible(ms3ggt_3,826). happensAt(walking(ms3ggt_3),826). orientation(ms3ggt_4,131,826). coords(ms3ggt_4,136,33,826).  visible(ms3ggt_4,826). happensAt(walking(ms3ggt_4),826).
orientation(ms3ggt_3,160,827). coords(ms3ggt_3,40,140,827).  visible(ms3ggt_3,827). happensAt(walking(ms3ggt_3),827). orientation(ms3ggt_4,131,827). coords(ms3ggt_4,136,33,827).  visible(ms3ggt_4,827). happensAt(walking(ms3ggt_4),827).
orientation(ms3ggt_3,160,828). coords(ms3ggt_3,40,141,828).  visible(ms3ggt_3,828). happensAt(walking(ms3ggt_3),828). orientation(ms3ggt_4,131,828). coords(ms3ggt_4,135,34,828).  visible(ms3ggt_4,828). happensAt(walking(ms3ggt_4),828).
orientation(ms3ggt_3,160,829). coords(ms3ggt_3,39,143,829).  visible(ms3ggt_3,829). happensAt(walking(ms3ggt_3),829). orientation(ms3ggt_4,131,829). coords(ms3ggt_4,135,34,829).  visible(ms3ggt_4,829). happensAt(walking(ms3ggt_4),829).
orientation(ms3ggt_3,156,830). coords(ms3ggt_3,39,143,830).  visible(ms3ggt_3,830). happensAt(walking(ms3ggt_3),830). orientation(ms3ggt_4,131,830). coords(ms3ggt_4,135,34,830).  visible(ms3ggt_4,830). happensAt(walking(ms3ggt_4),830).
orientation(ms3ggt_3,156,831). coords(ms3ggt_3,39,144,831).  visible(ms3ggt_3,831). happensAt(walking(ms3ggt_3),831). orientation(ms3ggt_4,136,831). coords(ms3ggt_4,134,34,831).  visible(ms3ggt_4,831). happensAt(walking(ms3ggt_4),831).
orientation(ms3ggt_3,156,832). coords(ms3ggt_3,39,144,832).  visible(ms3ggt_3,832). happensAt(walking(ms3ggt_3),832). orientation(ms3ggt_4,136,832). coords(ms3ggt_4,135,34,832).  visible(ms3ggt_4,832). happensAt(walking(ms3ggt_4),832).
orientation(ms3ggt_3,156,833). coords(ms3ggt_3,37,145,833).  visible(ms3ggt_3,833). happensAt(walking(ms3ggt_3),833). orientation(ms3ggt_4,136,833). coords(ms3ggt_4,135,34,833).  visible(ms3ggt_4,833). happensAt(walking(ms3ggt_4),833).
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
#show holdsAt/2.