
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

	orientation(ms3ggt_0,168,572). coords(ms3ggt_0,73,192,572).  visible(ms3ggt_0,572). happensAt(inactive(ms3ggt_0),572). orientation(ms3ggt_2,150,572). coords(ms3ggt_2,83,89,572).  visible(ms3ggt_2,572). happensAt(walking(ms3ggt_2),572).
orientation(ms3ggt_0,168,573). coords(ms3ggt_0,73,192,573).  visible(ms3ggt_0,573). happensAt(inactive(ms3ggt_0),573). orientation(ms3ggt_2,150,573). coords(ms3ggt_2,83,89,573).  visible(ms3ggt_2,573). happensAt(walking(ms3ggt_2),573).
orientation(ms3ggt_0,168,574). coords(ms3ggt_0,73,192,574).  visible(ms3ggt_0,574). happensAt(inactive(ms3ggt_0),574). orientation(ms3ggt_2,150,574). coords(ms3ggt_2,83,89,574).  visible(ms3ggt_2,574). happensAt(walking(ms3ggt_2),574).
orientation(ms3ggt_0,168,575). coords(ms3ggt_0,73,192,575).  visible(ms3ggt_0,575). happensAt(inactive(ms3ggt_0),575). orientation(ms3ggt_2,150,575). coords(ms3ggt_2,83,87,575).  visible(ms3ggt_2,575). happensAt(walking(ms3ggt_2),575).
orientation(ms3ggt_0,168,576). coords(ms3ggt_0,73,192,576).  visible(ms3ggt_0,576). happensAt(inactive(ms3ggt_0),576). orientation(ms3ggt_2,150,576). coords(ms3ggt_2,83,86,576).  visible(ms3ggt_2,576). happensAt(walking(ms3ggt_2),576).
orientation(ms3ggt_0,168,577). coords(ms3ggt_0,73,192,577).  visible(ms3ggt_0,577). happensAt(inactive(ms3ggt_0),577). orientation(ms3ggt_2,150,577). coords(ms3ggt_2,83,86,577).  visible(ms3ggt_2,577). happensAt(walking(ms3ggt_2),577).
orientation(ms3ggt_0,168,578). coords(ms3ggt_0,73,192,578).  visible(ms3ggt_0,578). happensAt(inactive(ms3ggt_0),578). orientation(ms3ggt_2,150,578). coords(ms3ggt_2,83,85,578).  visible(ms3ggt_2,578). happensAt(walking(ms3ggt_2),578).
orientation(ms3ggt_0,168,579). coords(ms3ggt_0,73,192,579).  visible(ms3ggt_0,579). happensAt(inactive(ms3ggt_0),579). orientation(ms3ggt_2,150,579). coords(ms3ggt_2,83,85,579).  visible(ms3ggt_2,579). happensAt(walking(ms3ggt_2),579).
orientation(ms3ggt_0,168,580). coords(ms3ggt_0,73,192,580).  visible(ms3ggt_0,580). happensAt(inactive(ms3ggt_0),580). orientation(ms3ggt_2,150,580). coords(ms3ggt_2,83,85,580).  visible(ms3ggt_2,580). happensAt(walking(ms3ggt_2),580).
orientation(ms3ggt_0,168,581). coords(ms3ggt_0,73,192,581).  visible(ms3ggt_0,581). happensAt(inactive(ms3ggt_0),581). orientation(ms3ggt_2,150,581). coords(ms3ggt_2,83,85,581).  visible(ms3ggt_2,581). happensAt(walking(ms3ggt_2),581).
orientation(ms3ggt_0,168,582). coords(ms3ggt_0,73,192,582).  visible(ms3ggt_0,582). happensAt(inactive(ms3ggt_0),582). orientation(ms3ggt_2,150,582). coords(ms3ggt_2,83,84,582).  visible(ms3ggt_2,582). happensAt(walking(ms3ggt_2),582).
orientation(ms3ggt_0,168,583). coords(ms3ggt_0,73,192,583).  visible(ms3ggt_0,583). happensAt(inactive(ms3ggt_0),583). orientation(ms3ggt_2,150,583). coords(ms3ggt_2,83,84,583).  visible(ms3ggt_2,583). happensAt(walking(ms3ggt_2),583).
orientation(ms3ggt_0,168,584). coords(ms3ggt_0,73,192,584).  visible(ms3ggt_0,584). happensAt(inactive(ms3ggt_0),584). orientation(ms3ggt_2,150,584). coords(ms3ggt_2,83,83,584).  visible(ms3ggt_2,584). happensAt(walking(ms3ggt_2),584).
orientation(ms3ggt_0,173,585). coords(ms3ggt_0,75,192,585).  visible(ms3ggt_0,585). happensAt(inactive(ms3ggt_0),585). orientation(ms3ggt_2,150,585). coords(ms3ggt_2,83,83,585).  visible(ms3ggt_2,585). happensAt(walking(ms3ggt_2),585).
orientation(ms3ggt_0,173,586). coords(ms3ggt_0,75,192,586).  visible(ms3ggt_0,586). happensAt(inactive(ms3ggt_0),586). orientation(ms3ggt_2,150,586). coords(ms3ggt_2,83,83,586).  visible(ms3ggt_2,586). happensAt(walking(ms3ggt_2),586).
orientation(ms3ggt_0,173,587). coords(ms3ggt_0,77,192,587).  visible(ms3ggt_0,587). happensAt(walking(ms3ggt_0),587). orientation(ms3ggt_2,150,587). coords(ms3ggt_2,83,83,587).  visible(ms3ggt_2,587). happensAt(walking(ms3ggt_2),587).
orientation(ms3ggt_0,173,588). coords(ms3ggt_0,77,192,588).  visible(ms3ggt_0,588). happensAt(walking(ms3ggt_0),588). orientation(ms3ggt_2,150,588). coords(ms3ggt_2,83,81,588).  visible(ms3ggt_2,588). happensAt(walking(ms3ggt_2),588).
orientation(ms3ggt_0,173,589). coords(ms3ggt_0,77,192,589).  visible(ms3ggt_0,589). happensAt(walking(ms3ggt_0),589). orientation(ms3ggt_2,150,589). coords(ms3ggt_2,83,81,589).  visible(ms3ggt_2,589). happensAt(walking(ms3ggt_2),589).
orientation(ms3ggt_0,173,590). coords(ms3ggt_0,77,192,590).  visible(ms3ggt_0,590). happensAt(walking(ms3ggt_0),590). orientation(ms3ggt_2,150,590). coords(ms3ggt_2,83,80,590).  visible(ms3ggt_2,590). happensAt(walking(ms3ggt_2),590).
orientation(ms3ggt_0,173,591). coords(ms3ggt_0,77,192,591).  visible(ms3ggt_0,591). happensAt(walking(ms3ggt_0),591). orientation(ms3ggt_2,150,591). coords(ms3ggt_2,83,80,591).  visible(ms3ggt_2,591). happensAt(walking(ms3ggt_2),591).
orientation(ms3ggt_0,173,592). coords(ms3ggt_0,77,192,592).  visible(ms3ggt_0,592). happensAt(walking(ms3ggt_0),592). orientation(ms3ggt_2,150,592). coords(ms3ggt_2,83,80,592).  visible(ms3ggt_2,592). happensAt(walking(ms3ggt_2),592).
orientation(ms3ggt_0,173,593). coords(ms3ggt_0,77,192,593).  visible(ms3ggt_0,593). happensAt(walking(ms3ggt_0),593). orientation(ms3ggt_2,150,593). coords(ms3ggt_2,83,80,593).  visible(ms3ggt_2,593). happensAt(walking(ms3ggt_2),593).
orientation(ms3ggt_0,164,594). coords(ms3ggt_0,78,192,594).  visible(ms3ggt_0,594). happensAt(walking(ms3ggt_0),594). orientation(ms3ggt_2,150,594). coords(ms3ggt_2,83,79,594).  visible(ms3ggt_2,594). happensAt(walking(ms3ggt_2),594).
orientation(ms3ggt_0,164,595). coords(ms3ggt_0,79,192,595).  visible(ms3ggt_0,595). happensAt(walking(ms3ggt_0),595). orientation(ms3ggt_2,150,595). coords(ms3ggt_2,83,79,595).  visible(ms3ggt_2,595). happensAt(walking(ms3ggt_2),595).
orientation(ms3ggt_0,164,596). coords(ms3ggt_0,79,192,596).  happensAt(appear(ms3ggt_0),596). happensAt(walking(ms3ggt_0),596). orientation(ms3ggt_2,150,596). coords(ms3ggt_2,83,79,596).  visible(ms3ggt_2,596). happensAt(walking(ms3ggt_2),596).
#show holdsAt/2.