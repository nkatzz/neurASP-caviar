
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

	orientation(lbgt_3,61,811). coords(lbgt_3,146,162,811).  visible(lbgt_3,811). happensAt(inactive(lbgt_3),811). orientation(lbgt_4,144,811). coords(lbgt_4,81,89,811).  visible(lbgt_4,811). happensAt(walking(lbgt_4),811).
orientation(lbgt_3,61,812). coords(lbgt_3,146,162,812).  visible(lbgt_3,812). happensAt(inactive(lbgt_3),812). orientation(lbgt_4,144,812). coords(lbgt_4,81,89,812).  visible(lbgt_4,812). happensAt(walking(lbgt_4),812).
orientation(lbgt_3,61,813). coords(lbgt_3,146,162,813).  visible(lbgt_3,813). happensAt(inactive(lbgt_3),813). orientation(lbgt_4,144,813). coords(lbgt_4,81,89,813).  visible(lbgt_4,813). happensAt(walking(lbgt_4),813).
orientation(lbgt_3,61,814). coords(lbgt_3,146,162,814).  visible(lbgt_3,814). happensAt(inactive(lbgt_3),814). orientation(lbgt_4,144,814). coords(lbgt_4,81,89,814).  visible(lbgt_4,814). happensAt(walking(lbgt_4),814).
orientation(lbgt_3,61,815). coords(lbgt_3,146,162,815).  visible(lbgt_3,815). happensAt(inactive(lbgt_3),815). orientation(lbgt_4,144,815). coords(lbgt_4,82,89,815).  visible(lbgt_4,815). happensAt(walking(lbgt_4),815).
orientation(lbgt_3,61,816). coords(lbgt_3,146,162,816).  visible(lbgt_3,816). happensAt(inactive(lbgt_3),816). orientation(lbgt_4,136,816). coords(lbgt_4,82,88,816).  visible(lbgt_4,816). happensAt(walking(lbgt_4),816).
orientation(lbgt_3,61,817). coords(lbgt_3,146,162,817).  visible(lbgt_3,817). happensAt(inactive(lbgt_3),817). orientation(lbgt_4,136,817). coords(lbgt_4,82,87,817).  visible(lbgt_4,817). happensAt(walking(lbgt_4),817).
orientation(lbgt_3,61,818). coords(lbgt_3,146,162,818).  visible(lbgt_3,818). happensAt(inactive(lbgt_3),818). orientation(lbgt_4,136,818). coords(lbgt_4,82,87,818).  visible(lbgt_4,818). happensAt(walking(lbgt_4),818).
orientation(lbgt_3,61,819). coords(lbgt_3,146,162,819).  visible(lbgt_3,819). happensAt(inactive(lbgt_3),819). orientation(lbgt_4,136,819). coords(lbgt_4,82,87,819).  visible(lbgt_4,819). happensAt(walking(lbgt_4),819).
orientation(lbgt_3,61,820). coords(lbgt_3,146,162,820).  visible(lbgt_3,820). happensAt(inactive(lbgt_3),820). orientation(lbgt_4,136,820). coords(lbgt_4,82,85,820).  visible(lbgt_4,820). happensAt(walking(lbgt_4),820).
orientation(lbgt_3,61,821). coords(lbgt_3,146,162,821).  visible(lbgt_3,821). happensAt(inactive(lbgt_3),821). orientation(lbgt_4,136,821). coords(lbgt_4,82,82,821).  visible(lbgt_4,821). happensAt(walking(lbgt_4),821).
orientation(lbgt_3,61,822). coords(lbgt_3,146,162,822).  visible(lbgt_3,822). happensAt(inactive(lbgt_3),822). orientation(lbgt_4,136,822). coords(lbgt_4,82,82,822).  visible(lbgt_4,822). happensAt(walking(lbgt_4),822).
orientation(lbgt_3,61,823). coords(lbgt_3,146,162,823).  visible(lbgt_3,823). happensAt(inactive(lbgt_3),823). orientation(lbgt_4,136,823). coords(lbgt_4,82,82,823).  visible(lbgt_4,823). happensAt(walking(lbgt_4),823).
orientation(lbgt_3,61,824). coords(lbgt_3,146,162,824).  visible(lbgt_3,824). happensAt(inactive(lbgt_3),824). orientation(lbgt_4,136,824). coords(lbgt_4,82,82,824).  visible(lbgt_4,824). happensAt(walking(lbgt_4),824).
orientation(lbgt_3,61,825). coords(lbgt_3,146,162,825).  visible(lbgt_3,825). happensAt(inactive(lbgt_3),825). orientation(lbgt_4,136,825). coords(lbgt_4,82,82,825).  visible(lbgt_4,825). happensAt(walking(lbgt_4),825).
orientation(lbgt_3,61,826). coords(lbgt_3,146,162,826).  visible(lbgt_3,826). happensAt(inactive(lbgt_3),826). orientation(lbgt_4,136,826). coords(lbgt_4,82,82,826).  visible(lbgt_4,826). happensAt(walking(lbgt_4),826).
orientation(lbgt_3,61,827). coords(lbgt_3,146,162,827).  visible(lbgt_3,827). happensAt(inactive(lbgt_3),827). orientation(lbgt_4,136,827). coords(lbgt_4,82,81,827).  visible(lbgt_4,827). happensAt(walking(lbgt_4),827).
orientation(lbgt_3,61,828). coords(lbgt_3,146,162,828).  visible(lbgt_3,828). happensAt(inactive(lbgt_3),828). orientation(lbgt_4,136,828). coords(lbgt_4,82,80,828).  visible(lbgt_4,828). happensAt(walking(lbgt_4),828).
orientation(lbgt_3,61,829). coords(lbgt_3,146,162,829).  visible(lbgt_3,829). happensAt(inactive(lbgt_3),829). orientation(lbgt_4,136,829). coords(lbgt_4,82,79,829).  visible(lbgt_4,829). happensAt(walking(lbgt_4),829).
orientation(lbgt_3,61,830). coords(lbgt_3,146,162,830).  visible(lbgt_3,830). happensAt(inactive(lbgt_3),830). orientation(lbgt_4,136,830). coords(lbgt_4,82,79,830).  visible(lbgt_4,830). happensAt(walking(lbgt_4),830).
orientation(lbgt_3,61,831). coords(lbgt_3,146,162,831).  visible(lbgt_3,831). happensAt(inactive(lbgt_3),831). orientation(lbgt_4,136,831). coords(lbgt_4,82,78,831).  visible(lbgt_4,831). happensAt(walking(lbgt_4),831).
orientation(lbgt_3,61,832). coords(lbgt_3,146,162,832).  visible(lbgt_3,832). happensAt(inactive(lbgt_3),832). orientation(lbgt_4,136,832). coords(lbgt_4,82,76,832).  visible(lbgt_4,832). happensAt(walking(lbgt_4),832).
orientation(lbgt_3,61,833). coords(lbgt_3,146,162,833).  visible(lbgt_3,833). happensAt(inactive(lbgt_3),833). orientation(lbgt_4,136,833). coords(lbgt_4,82,75,833).  visible(lbgt_4,833). happensAt(walking(lbgt_4),833).
orientation(lbgt_3,61,834). coords(lbgt_3,146,162,834).  visible(lbgt_3,834). happensAt(inactive(lbgt_3),834). orientation(lbgt_4,136,834). coords(lbgt_4,82,75,834).  visible(lbgt_4,834). happensAt(walking(lbgt_4),834).
orientation(lbgt_3,61,835). coords(lbgt_3,146,162,835).  visible(lbgt_3,835). happensAt(inactive(lbgt_3),835). orientation(lbgt_4,136,835). coords(lbgt_4,82,75,835).  visible(lbgt_4,835). happensAt(walking(lbgt_4),835).
#show holdsAt/2.