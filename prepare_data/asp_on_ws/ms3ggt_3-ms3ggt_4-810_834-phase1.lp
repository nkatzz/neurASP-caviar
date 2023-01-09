
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

	orientation(ms3ggt_3,147,810). coords(ms3ggt_3,44,121,810).  visible(ms3ggt_3,810). happensAt(walking(ms3ggt_3),810). orientation(ms3ggt_4,120,810). coords(ms3ggt_4,138,30,810).  visible(ms3ggt_4,810). happensAt(walking(ms3ggt_4),810).
orientation(ms3ggt_3,147,811). coords(ms3ggt_3,44,122,811).  visible(ms3ggt_3,811). happensAt(walking(ms3ggt_3),811). orientation(ms3ggt_4,120,811). coords(ms3ggt_4,138,32,811).  visible(ms3ggt_4,811). happensAt(walking(ms3ggt_4),811).
orientation(ms3ggt_3,147,812). coords(ms3ggt_3,44,126,812).  visible(ms3ggt_3,812). happensAt(walking(ms3ggt_3),812). orientation(ms3ggt_4,131,812). coords(ms3ggt_4,138,32,812).  visible(ms3ggt_4,812). happensAt(walking(ms3ggt_4),812).
orientation(ms3ggt_3,147,813). coords(ms3ggt_3,44,126,813).  visible(ms3ggt_3,813). happensAt(walking(ms3ggt_3),813). orientation(ms3ggt_4,131,813). coords(ms3ggt_4,138,32,813).  visible(ms3ggt_4,813). happensAt(walking(ms3ggt_4),813).
orientation(ms3ggt_3,147,814). coords(ms3ggt_3,44,127,814).  visible(ms3ggt_3,814). happensAt(walking(ms3ggt_3),814). orientation(ms3ggt_4,131,814). coords(ms3ggt_4,137,32,814).  visible(ms3ggt_4,814). happensAt(walking(ms3ggt_4),814).
orientation(ms3ggt_3,147,815). coords(ms3ggt_3,44,127,815).  visible(ms3ggt_3,815). happensAt(walking(ms3ggt_3),815). orientation(ms3ggt_4,131,815). coords(ms3ggt_4,137,32,815).  visible(ms3ggt_4,815). happensAt(walking(ms3ggt_4),815).
orientation(ms3ggt_3,147,816). coords(ms3ggt_3,44,128,816).  visible(ms3ggt_3,816). happensAt(walking(ms3ggt_3),816). orientation(ms3ggt_4,131,816). coords(ms3ggt_4,137,32,816).  visible(ms3ggt_4,816). happensAt(walking(ms3ggt_4),816).
orientation(ms3ggt_3,147,817). coords(ms3ggt_3,44,129,817).  visible(ms3ggt_3,817). happensAt(walking(ms3ggt_3),817). orientation(ms3ggt_4,131,817). coords(ms3ggt_4,137,32,817).  visible(ms3ggt_4,817). happensAt(walking(ms3ggt_4),817).
orientation(ms3ggt_3,153,818). coords(ms3ggt_3,44,129,818).  visible(ms3ggt_3,818). happensAt(walking(ms3ggt_3),818). orientation(ms3ggt_4,131,818). coords(ms3ggt_4,136,32,818).  visible(ms3ggt_4,818). happensAt(walking(ms3ggt_4),818).
orientation(ms3ggt_3,153,819). coords(ms3ggt_3,42,131,819).  visible(ms3ggt_3,819). happensAt(walking(ms3ggt_3),819). orientation(ms3ggt_4,131,819). coords(ms3ggt_4,136,32,819).  visible(ms3ggt_4,819). happensAt(walking(ms3ggt_4),819).
orientation(ms3ggt_3,153,820). coords(ms3ggt_3,42,132,820).  visible(ms3ggt_3,820). happensAt(walking(ms3ggt_3),820). orientation(ms3ggt_4,131,820). coords(ms3ggt_4,136,32,820).  visible(ms3ggt_4,820). happensAt(walking(ms3ggt_4),820).
orientation(ms3ggt_3,153,821). coords(ms3ggt_3,42,132,821).  visible(ms3ggt_3,821). happensAt(walking(ms3ggt_3),821). orientation(ms3ggt_4,131,821). coords(ms3ggt_4,136,32,821).  visible(ms3ggt_4,821). happensAt(walking(ms3ggt_4),821).
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
#show holdsAt/2.