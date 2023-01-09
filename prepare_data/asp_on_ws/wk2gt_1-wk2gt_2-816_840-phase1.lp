
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

	orientation(wk2gt_1,125,816). coords(wk2gt_1,115,53,816).  visible(wk2gt_1,816). happensAt(inactive(wk2gt_1),816). orientation(wk2gt_2,142,816). coords(wk2gt_2,99,63,816).  visible(wk2gt_2,816). happensAt(inactive(wk2gt_2),816).
orientation(wk2gt_1,125,817). coords(wk2gt_1,115,53,817).  visible(wk2gt_1,817). happensAt(inactive(wk2gt_1),817). orientation(wk2gt_2,142,817). coords(wk2gt_2,99,63,817).  visible(wk2gt_2,817). happensAt(inactive(wk2gt_2),817).
orientation(wk2gt_1,125,818). coords(wk2gt_1,115,53,818).  visible(wk2gt_1,818). happensAt(inactive(wk2gt_1),818). orientation(wk2gt_2,142,818). coords(wk2gt_2,99,63,818).  visible(wk2gt_2,818). happensAt(inactive(wk2gt_2),818).
orientation(wk2gt_1,125,819). coords(wk2gt_1,115,53,819).  visible(wk2gt_1,819). happensAt(inactive(wk2gt_1),819). orientation(wk2gt_2,142,819). coords(wk2gt_2,99,63,819).  visible(wk2gt_2,819). happensAt(inactive(wk2gt_2),819).
orientation(wk2gt_1,125,820). coords(wk2gt_1,115,53,820).  visible(wk2gt_1,820). happensAt(inactive(wk2gt_1),820). orientation(wk2gt_2,142,820). coords(wk2gt_2,99,63,820).  visible(wk2gt_2,820). happensAt(inactive(wk2gt_2),820).
orientation(wk2gt_1,125,821). coords(wk2gt_1,115,53,821).  visible(wk2gt_1,821). happensAt(inactive(wk2gt_1),821). orientation(wk2gt_2,142,821). coords(wk2gt_2,99,63,821).  visible(wk2gt_2,821). happensAt(inactive(wk2gt_2),821).
orientation(wk2gt_1,125,822). coords(wk2gt_1,115,53,822).  visible(wk2gt_1,822). happensAt(inactive(wk2gt_1),822). orientation(wk2gt_2,142,822). coords(wk2gt_2,99,63,822).  visible(wk2gt_2,822). happensAt(inactive(wk2gt_2),822).
orientation(wk2gt_1,125,823). coords(wk2gt_1,115,53,823).  visible(wk2gt_1,823). happensAt(inactive(wk2gt_1),823). orientation(wk2gt_2,142,823). coords(wk2gt_2,99,63,823).  visible(wk2gt_2,823). happensAt(inactive(wk2gt_2),823).
orientation(wk2gt_1,125,824). coords(wk2gt_1,115,53,824).  visible(wk2gt_1,824). happensAt(inactive(wk2gt_1),824). orientation(wk2gt_2,142,824). coords(wk2gt_2,99,63,824).  visible(wk2gt_2,824). happensAt(inactive(wk2gt_2),824).
orientation(wk2gt_1,125,825). coords(wk2gt_1,115,53,825).  visible(wk2gt_1,825). happensAt(inactive(wk2gt_1),825). orientation(wk2gt_2,142,825). coords(wk2gt_2,99,63,825).  visible(wk2gt_2,825). happensAt(inactive(wk2gt_2),825).
orientation(wk2gt_1,125,826). coords(wk2gt_1,115,53,826).  visible(wk2gt_1,826). happensAt(inactive(wk2gt_1),826). orientation(wk2gt_2,142,826). coords(wk2gt_2,99,63,826).  visible(wk2gt_2,826). happensAt(inactive(wk2gt_2),826).
orientation(wk2gt_1,125,827). coords(wk2gt_1,115,53,827).  visible(wk2gt_1,827). happensAt(inactive(wk2gt_1),827). orientation(wk2gt_2,142,827). coords(wk2gt_2,99,63,827).  visible(wk2gt_2,827). happensAt(inactive(wk2gt_2),827).
orientation(wk2gt_1,125,828). coords(wk2gt_1,115,53,828).  visible(wk2gt_1,828). happensAt(inactive(wk2gt_1),828). orientation(wk2gt_2,142,828). coords(wk2gt_2,99,63,828).  visible(wk2gt_2,828). happensAt(inactive(wk2gt_2),828).
orientation(wk2gt_1,125,829). coords(wk2gt_1,115,53,829).  visible(wk2gt_1,829). happensAt(inactive(wk2gt_1),829). orientation(wk2gt_2,142,829). coords(wk2gt_2,99,63,829).  visible(wk2gt_2,829). happensAt(inactive(wk2gt_2),829).
orientation(wk2gt_1,125,830). coords(wk2gt_1,115,53,830).  visible(wk2gt_1,830). happensAt(inactive(wk2gt_1),830). orientation(wk2gt_2,142,830). coords(wk2gt_2,99,63,830).  visible(wk2gt_2,830). happensAt(inactive(wk2gt_2),830).
orientation(wk2gt_1,125,831). coords(wk2gt_1,115,53,831).  visible(wk2gt_1,831). happensAt(inactive(wk2gt_1),831). orientation(wk2gt_2,142,831). coords(wk2gt_2,99,63,831).  visible(wk2gt_2,831). happensAt(inactive(wk2gt_2),831).
orientation(wk2gt_1,125,832). coords(wk2gt_1,115,53,832).  visible(wk2gt_1,832). happensAt(inactive(wk2gt_1),832). orientation(wk2gt_2,142,832). coords(wk2gt_2,99,63,832).  visible(wk2gt_2,832). happensAt(inactive(wk2gt_2),832).
orientation(wk2gt_1,125,833). coords(wk2gt_1,115,53,833).  visible(wk2gt_1,833). happensAt(inactive(wk2gt_1),833). orientation(wk2gt_2,142,833). coords(wk2gt_2,99,63,833).  visible(wk2gt_2,833). happensAt(inactive(wk2gt_2),833).
orientation(wk2gt_1,125,834). coords(wk2gt_1,115,53,834).  visible(wk2gt_1,834). happensAt(inactive(wk2gt_1),834). orientation(wk2gt_2,142,834). coords(wk2gt_2,99,63,834).  visible(wk2gt_2,834). happensAt(inactive(wk2gt_2),834).
orientation(wk2gt_1,125,835). coords(wk2gt_1,115,53,835).  visible(wk2gt_1,835). happensAt(inactive(wk2gt_1),835). orientation(wk2gt_2,142,835). coords(wk2gt_2,99,63,835).  visible(wk2gt_2,835). happensAt(inactive(wk2gt_2),835).
orientation(wk2gt_1,125,836). coords(wk2gt_1,115,53,836).  visible(wk2gt_1,836). happensAt(inactive(wk2gt_1),836). orientation(wk2gt_2,142,836). coords(wk2gt_2,99,63,836).  visible(wk2gt_2,836). happensAt(inactive(wk2gt_2),836).
orientation(wk2gt_1,125,837). coords(wk2gt_1,115,53,837).  visible(wk2gt_1,837). happensAt(inactive(wk2gt_1),837). orientation(wk2gt_2,142,837). coords(wk2gt_2,99,63,837).  visible(wk2gt_2,837). happensAt(inactive(wk2gt_2),837).
orientation(wk2gt_1,125,838). coords(wk2gt_1,115,53,838).  visible(wk2gt_1,838). happensAt(inactive(wk2gt_1),838). orientation(wk2gt_2,142,838). coords(wk2gt_2,99,63,838).  visible(wk2gt_2,838). happensAt(inactive(wk2gt_2),838).
orientation(wk2gt_1,125,839). coords(wk2gt_1,115,53,839).  visible(wk2gt_1,839). happensAt(inactive(wk2gt_1),839). orientation(wk2gt_2,142,839). coords(wk2gt_2,99,63,839).  visible(wk2gt_2,839). happensAt(inactive(wk2gt_2),839).
orientation(wk2gt_1,125,840). coords(wk2gt_1,115,53,840).  visible(wk2gt_1,840). happensAt(inactive(wk2gt_1),840). orientation(wk2gt_2,142,840). coords(wk2gt_2,99,63,840).  visible(wk2gt_2,840). happensAt(inactive(wk2gt_2),840).
#show holdsAt/2.