
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

	orientation(lbpugt_2,132,575). coords(lbpugt_2,181,50,575).  visible(lbpugt_2,575). happensAt(inactive(lbpugt_2),575). orientation(lbpugt_4,135,575). coords(lbpugt_4,219,34,575).  visible(lbpugt_4,575). happensAt(inactive(lbpugt_4),575).
orientation(lbpugt_2,132,576). coords(lbpugt_2,181,50,576).  visible(lbpugt_2,576). happensAt(inactive(lbpugt_2),576). orientation(lbpugt_4,135,576). coords(lbpugt_4,219,34,576).  visible(lbpugt_4,576). happensAt(inactive(lbpugt_4),576).
orientation(lbpugt_2,132,577). coords(lbpugt_2,181,50,577).  visible(lbpugt_2,577). happensAt(inactive(lbpugt_2),577). orientation(lbpugt_4,135,577). coords(lbpugt_4,219,34,577).  visible(lbpugt_4,577). happensAt(inactive(lbpugt_4),577).
orientation(lbpugt_2,132,578). coords(lbpugt_2,181,50,578).  visible(lbpugt_2,578). happensAt(inactive(lbpugt_2),578). orientation(lbpugt_4,135,578). coords(lbpugt_4,219,34,578).  visible(lbpugt_4,578). happensAt(inactive(lbpugt_4),578).
orientation(lbpugt_2,132,579). coords(lbpugt_2,181,50,579).  visible(lbpugt_2,579). happensAt(inactive(lbpugt_2),579). orientation(lbpugt_4,135,579). coords(lbpugt_4,219,34,579).  visible(lbpugt_4,579). happensAt(inactive(lbpugt_4),579).
orientation(lbpugt_2,132,580). coords(lbpugt_2,181,50,580).  visible(lbpugt_2,580). happensAt(inactive(lbpugt_2),580). orientation(lbpugt_4,135,580). coords(lbpugt_4,219,34,580).  visible(lbpugt_4,580). happensAt(inactive(lbpugt_4),580).
orientation(lbpugt_2,132,581). coords(lbpugt_2,181,50,581).  visible(lbpugt_2,581). happensAt(inactive(lbpugt_2),581). orientation(lbpugt_4,135,581). coords(lbpugt_4,219,34,581).  visible(lbpugt_4,581). happensAt(inactive(lbpugt_4),581).
orientation(lbpugt_2,132,582). coords(lbpugt_2,181,50,582).  visible(lbpugt_2,582). happensAt(inactive(lbpugt_2),582). orientation(lbpugt_4,135,582). coords(lbpugt_4,219,34,582).  visible(lbpugt_4,582). happensAt(inactive(lbpugt_4),582).
orientation(lbpugt_2,132,583). coords(lbpugt_2,181,50,583).  visible(lbpugt_2,583). happensAt(inactive(lbpugt_2),583). orientation(lbpugt_4,135,583). coords(lbpugt_4,219,34,583).  visible(lbpugt_4,583). happensAt(inactive(lbpugt_4),583).
orientation(lbpugt_2,132,584). coords(lbpugt_2,181,50,584).  visible(lbpugt_2,584). happensAt(inactive(lbpugt_2),584). orientation(lbpugt_4,135,584). coords(lbpugt_4,219,34,584).  visible(lbpugt_4,584). happensAt(inactive(lbpugt_4),584).
orientation(lbpugt_2,132,585). coords(lbpugt_2,181,50,585).  visible(lbpugt_2,585). happensAt(inactive(lbpugt_2),585). orientation(lbpugt_4,135,585). coords(lbpugt_4,219,34,585).  visible(lbpugt_4,585). happensAt(inactive(lbpugt_4),585).
orientation(lbpugt_2,132,586). coords(lbpugt_2,181,50,586).  visible(lbpugt_2,586). happensAt(inactive(lbpugt_2),586). orientation(lbpugt_4,135,586). coords(lbpugt_4,219,34,586).  visible(lbpugt_4,586). happensAt(inactive(lbpugt_4),586).
orientation(lbpugt_2,132,587). coords(lbpugt_2,181,50,587).  visible(lbpugt_2,587). happensAt(inactive(lbpugt_2),587). orientation(lbpugt_4,135,587). coords(lbpugt_4,219,34,587).  visible(lbpugt_4,587). happensAt(inactive(lbpugt_4),587).
orientation(lbpugt_2,132,588). coords(lbpugt_2,181,50,588).  visible(lbpugt_2,588). happensAt(inactive(lbpugt_2),588). orientation(lbpugt_4,135,588). coords(lbpugt_4,219,34,588).  visible(lbpugt_4,588). happensAt(inactive(lbpugt_4),588).
orientation(lbpugt_2,132,589). coords(lbpugt_2,181,50,589).  visible(lbpugt_2,589). happensAt(inactive(lbpugt_2),589). orientation(lbpugt_4,135,589). coords(lbpugt_4,219,34,589).  visible(lbpugt_4,589). happensAt(inactive(lbpugt_4),589).
orientation(lbpugt_2,132,590). coords(lbpugt_2,181,50,590).  visible(lbpugt_2,590). happensAt(inactive(lbpugt_2),590). orientation(lbpugt_4,135,590). coords(lbpugt_4,219,34,590).  visible(lbpugt_4,590). happensAt(inactive(lbpugt_4),590).
orientation(lbpugt_2,132,591). coords(lbpugt_2,181,50,591).  visible(lbpugt_2,591). happensAt(inactive(lbpugt_2),591). orientation(lbpugt_4,135,591). coords(lbpugt_4,219,34,591).  visible(lbpugt_4,591). happensAt(inactive(lbpugt_4),591).
orientation(lbpugt_2,132,592). coords(lbpugt_2,181,50,592).  visible(lbpugt_2,592). happensAt(inactive(lbpugt_2),592). orientation(lbpugt_4,135,592). coords(lbpugt_4,219,34,592).  visible(lbpugt_4,592). happensAt(inactive(lbpugt_4),592).
orientation(lbpugt_2,132,593). coords(lbpugt_2,181,50,593).  visible(lbpugt_2,593). happensAt(inactive(lbpugt_2),593). orientation(lbpugt_4,135,593). coords(lbpugt_4,219,34,593).  visible(lbpugt_4,593). happensAt(inactive(lbpugt_4),593).
orientation(lbpugt_2,132,594). coords(lbpugt_2,181,50,594).  visible(lbpugt_2,594). happensAt(inactive(lbpugt_2),594). orientation(lbpugt_4,135,594). coords(lbpugt_4,219,34,594).  visible(lbpugt_4,594). happensAt(inactive(lbpugt_4),594).
orientation(lbpugt_2,132,595). coords(lbpugt_2,181,50,595).  visible(lbpugt_2,595). happensAt(inactive(lbpugt_2),595). orientation(lbpugt_4,135,595). coords(lbpugt_4,219,34,595).  visible(lbpugt_4,595). happensAt(inactive(lbpugt_4),595).
orientation(lbpugt_2,132,596). coords(lbpugt_2,181,50,596).  visible(lbpugt_2,596). happensAt(inactive(lbpugt_2),596). orientation(lbpugt_4,135,596). coords(lbpugt_4,219,34,596).  visible(lbpugt_4,596). happensAt(inactive(lbpugt_4),596).
orientation(lbpugt_2,132,597). coords(lbpugt_2,181,50,597).  visible(lbpugt_2,597). happensAt(inactive(lbpugt_2),597). orientation(lbpugt_4,135,597). coords(lbpugt_4,219,34,597).  visible(lbpugt_4,597). happensAt(inactive(lbpugt_4),597).
orientation(lbpugt_2,132,598). coords(lbpugt_2,181,50,598).  visible(lbpugt_2,598). happensAt(inactive(lbpugt_2),598). orientation(lbpugt_4,135,598). coords(lbpugt_4,219,34,598).  visible(lbpugt_4,598). happensAt(inactive(lbpugt_4),598).
orientation(lbpugt_2,132,599). coords(lbpugt_2,181,50,599).  visible(lbpugt_2,599). happensAt(inactive(lbpugt_2),599). orientation(lbpugt_4,135,599). coords(lbpugt_4,219,34,599).  visible(lbpugt_4,599). happensAt(inactive(lbpugt_4),599).
#show holdsAt/2.