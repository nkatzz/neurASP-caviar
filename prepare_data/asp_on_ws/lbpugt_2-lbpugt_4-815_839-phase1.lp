
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

	orientation(lbpugt_2,91,815). coords(lbpugt_2,294,102,815).  visible(lbpugt_2,815). happensAt(active(lbpugt_2),815). orientation(lbpugt_4,135,815). coords(lbpugt_4,219,34,815).  visible(lbpugt_4,815). happensAt(inactive(lbpugt_4),815).
orientation(lbpugt_2,91,816). coords(lbpugt_2,294,101,816).  visible(lbpugt_2,816). happensAt(active(lbpugt_2),816). orientation(lbpugt_4,135,816). coords(lbpugt_4,219,34,816).  visible(lbpugt_4,816). happensAt(inactive(lbpugt_4),816).
orientation(lbpugt_2,89,817). coords(lbpugt_2,294,101,817).  visible(lbpugt_2,817). happensAt(active(lbpugt_2),817). orientation(lbpugt_4,135,817). coords(lbpugt_4,219,34,817).  visible(lbpugt_4,817). happensAt(inactive(lbpugt_4),817).
orientation(lbpugt_2,89,818). coords(lbpugt_2,294,100,818).  visible(lbpugt_2,818). happensAt(active(lbpugt_2),818). orientation(lbpugt_4,135,818). coords(lbpugt_4,219,34,818).  visible(lbpugt_4,818). happensAt(inactive(lbpugt_4),818).
orientation(lbpugt_2,89,819). coords(lbpugt_2,294,100,819).  visible(lbpugt_2,819). happensAt(active(lbpugt_2),819). orientation(lbpugt_4,135,819). coords(lbpugt_4,219,34,819).  visible(lbpugt_4,819). happensAt(inactive(lbpugt_4),819).
orientation(lbpugt_2,89,820). coords(lbpugt_2,293,100,820).  visible(lbpugt_2,820). happensAt(active(lbpugt_2),820). orientation(lbpugt_4,135,820). coords(lbpugt_4,219,34,820).  visible(lbpugt_4,820). happensAt(inactive(lbpugt_4),820).
orientation(lbpugt_2,89,821). coords(lbpugt_2,293,100,821).  visible(lbpugt_2,821). happensAt(active(lbpugt_2),821). orientation(lbpugt_4,135,821). coords(lbpugt_4,219,34,821).  visible(lbpugt_4,821). happensAt(inactive(lbpugt_4),821).
orientation(lbpugt_2,89,822). coords(lbpugt_2,292,100,822).  visible(lbpugt_2,822). happensAt(active(lbpugt_2),822). orientation(lbpugt_4,135,822). coords(lbpugt_4,219,34,822).  visible(lbpugt_4,822). happensAt(inactive(lbpugt_4),822).
orientation(lbpugt_2,89,823). coords(lbpugt_2,292,100,823).  visible(lbpugt_2,823). happensAt(active(lbpugt_2),823). orientation(lbpugt_4,135,823). coords(lbpugt_4,219,34,823).  visible(lbpugt_4,823). happensAt(inactive(lbpugt_4),823).
orientation(lbpugt_2,89,824). coords(lbpugt_2,292,100,824).  visible(lbpugt_2,824). happensAt(active(lbpugt_2),824). orientation(lbpugt_4,135,824). coords(lbpugt_4,219,34,824).  visible(lbpugt_4,824). happensAt(inactive(lbpugt_4),824).
orientation(lbpugt_2,89,825). coords(lbpugt_2,292,100,825).  visible(lbpugt_2,825). happensAt(active(lbpugt_2),825). orientation(lbpugt_4,135,825). coords(lbpugt_4,219,34,825).  visible(lbpugt_4,825). happensAt(inactive(lbpugt_4),825).
orientation(lbpugt_2,89,826). coords(lbpugt_2,292,100,826).  visible(lbpugt_2,826). happensAt(active(lbpugt_2),826). orientation(lbpugt_4,135,826). coords(lbpugt_4,219,34,826).  visible(lbpugt_4,826). happensAt(inactive(lbpugt_4),826).
orientation(lbpugt_2,86,827). coords(lbpugt_2,292,100,827).  visible(lbpugt_2,827). happensAt(active(lbpugt_2),827). orientation(lbpugt_4,135,827). coords(lbpugt_4,219,34,827).  visible(lbpugt_4,827). happensAt(inactive(lbpugt_4),827).
orientation(lbpugt_2,86,828). coords(lbpugt_2,292,100,828).  visible(lbpugt_2,828). happensAt(active(lbpugt_2),828). orientation(lbpugt_4,135,828). coords(lbpugt_4,219,34,828).  visible(lbpugt_4,828). happensAt(inactive(lbpugt_4),828).
orientation(lbpugt_2,85,829). coords(lbpugt_2,292,100,829).  visible(lbpugt_2,829). happensAt(active(lbpugt_2),829). orientation(lbpugt_4,135,829). coords(lbpugt_4,219,34,829).  visible(lbpugt_4,829). happensAt(inactive(lbpugt_4),829).
orientation(lbpugt_2,85,830). coords(lbpugt_2,292,99,830).  visible(lbpugt_2,830). happensAt(active(lbpugt_2),830). orientation(lbpugt_4,135,830). coords(lbpugt_4,219,34,830).  visible(lbpugt_4,830). happensAt(inactive(lbpugt_4),830).
orientation(lbpugt_2,84,831). coords(lbpugt_2,292,99,831).  visible(lbpugt_2,831). happensAt(active(lbpugt_2),831). orientation(lbpugt_4,135,831). coords(lbpugt_4,219,34,831).  visible(lbpugt_4,831). happensAt(inactive(lbpugt_4),831).
orientation(lbpugt_2,84,832). coords(lbpugt_2,292,99,832).  visible(lbpugt_2,832). happensAt(active(lbpugt_2),832). orientation(lbpugt_4,135,832). coords(lbpugt_4,219,34,832).  visible(lbpugt_4,832). happensAt(inactive(lbpugt_4),832).
orientation(lbpugt_2,84,833). coords(lbpugt_2,292,98,833).  visible(lbpugt_2,833). happensAt(active(lbpugt_2),833). orientation(lbpugt_4,135,833). coords(lbpugt_4,219,34,833).  visible(lbpugt_4,833). happensAt(inactive(lbpugt_4),833).
orientation(lbpugt_2,84,834). coords(lbpugt_2,292,98,834).  visible(lbpugt_2,834). happensAt(active(lbpugt_2),834). orientation(lbpugt_4,135,834). coords(lbpugt_4,219,34,834).  visible(lbpugt_4,834). happensAt(inactive(lbpugt_4),834).
orientation(lbpugt_2,84,835). coords(lbpugt_2,292,97,835).  visible(lbpugt_2,835). happensAt(active(lbpugt_2),835). orientation(lbpugt_4,135,835). coords(lbpugt_4,219,34,835).  visible(lbpugt_4,835). happensAt(inactive(lbpugt_4),835).
orientation(lbpugt_2,84,836). coords(lbpugt_2,292,97,836).  visible(lbpugt_2,836). happensAt(active(lbpugt_2),836). orientation(lbpugt_4,135,836). coords(lbpugt_4,219,34,836).  visible(lbpugt_4,836). happensAt(inactive(lbpugt_4),836).
orientation(lbpugt_2,84,837). coords(lbpugt_2,292,96,837).  visible(lbpugt_2,837). happensAt(active(lbpugt_2),837). orientation(lbpugt_4,135,837). coords(lbpugt_4,219,34,837).  visible(lbpugt_4,837). happensAt(inactive(lbpugt_4),837).
orientation(lbpugt_2,84,838). coords(lbpugt_2,292,96,838).  visible(lbpugt_2,838). happensAt(active(lbpugt_2),838). orientation(lbpugt_4,135,838). coords(lbpugt_4,219,34,838).  visible(lbpugt_4,838). happensAt(inactive(lbpugt_4),838).
orientation(lbpugt_2,84,839). coords(lbpugt_2,292,96,839).  visible(lbpugt_2,839). happensAt(active(lbpugt_2),839). orientation(lbpugt_4,135,839). coords(lbpugt_4,219,34,839).  visible(lbpugt_4,839). happensAt(inactive(lbpugt_4),839).
#show holdsAt/2.