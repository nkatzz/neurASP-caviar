
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

	orientation(lb2gt_4,154,575). coords(lb2gt_4,133,42,575).  visible(lb2gt_4,575). happensAt(walking(lb2gt_4),575). orientation(lb2gt_6,151,575). coords(lb2gt_6,72,122,575).  visible(lb2gt_6,575). happensAt(inactive(lb2gt_6),575).
orientation(lb2gt_4,154,576). coords(lb2gt_4,134,41,576).  visible(lb2gt_4,576). happensAt(walking(lb2gt_4),576). orientation(lb2gt_6,151,576). coords(lb2gt_6,72,122,576).  visible(lb2gt_6,576). happensAt(inactive(lb2gt_6),576).
orientation(lb2gt_4,154,577). coords(lb2gt_4,134,40,577).  visible(lb2gt_4,577). happensAt(walking(lb2gt_4),577). orientation(lb2gt_6,151,577). coords(lb2gt_6,72,122,577).  visible(lb2gt_6,577). happensAt(inactive(lb2gt_6),577).
orientation(lb2gt_4,154,578). coords(lb2gt_4,134,40,578).  visible(lb2gt_4,578). happensAt(walking(lb2gt_4),578). orientation(lb2gt_6,151,578). coords(lb2gt_6,72,122,578).  visible(lb2gt_6,578). happensAt(inactive(lb2gt_6),578).
orientation(lb2gt_4,154,579). coords(lb2gt_4,134,39,579).  visible(lb2gt_4,579). happensAt(walking(lb2gt_4),579). orientation(lb2gt_6,151,579). coords(lb2gt_6,72,122,579).  visible(lb2gt_6,579). happensAt(inactive(lb2gt_6),579).
orientation(lb2gt_4,154,580). coords(lb2gt_4,136,38,580).  visible(lb2gt_4,580). happensAt(walking(lb2gt_4),580). orientation(lb2gt_6,151,580). coords(lb2gt_6,72,122,580).  visible(lb2gt_6,580). happensAt(inactive(lb2gt_6),580).
orientation(lb2gt_4,154,581). coords(lb2gt_4,137,38,581).  visible(lb2gt_4,581). happensAt(walking(lb2gt_4),581). orientation(lb2gt_6,151,581). coords(lb2gt_6,72,122,581).  visible(lb2gt_6,581). happensAt(inactive(lb2gt_6),581).
orientation(lb2gt_4,154,582). coords(lb2gt_4,138,38,582).  visible(lb2gt_4,582). happensAt(walking(lb2gt_4),582). orientation(lb2gt_6,151,582). coords(lb2gt_6,72,122,582).  visible(lb2gt_6,582). happensAt(inactive(lb2gt_6),582).
orientation(lb2gt_4,154,583). coords(lb2gt_4,139,38,583).  visible(lb2gt_4,583). happensAt(walking(lb2gt_4),583). orientation(lb2gt_6,151,583). coords(lb2gt_6,72,122,583).  visible(lb2gt_6,583). happensAt(inactive(lb2gt_6),583).
orientation(lb2gt_4,154,584). coords(lb2gt_4,140,37,584).  visible(lb2gt_4,584). happensAt(walking(lb2gt_4),584). orientation(lb2gt_6,151,584). coords(lb2gt_6,72,122,584).  visible(lb2gt_6,584). happensAt(inactive(lb2gt_6),584).
orientation(lb2gt_4,154,585). coords(lb2gt_4,140,37,585).  visible(lb2gt_4,585). happensAt(walking(lb2gt_4),585). orientation(lb2gt_6,151,585). coords(lb2gt_6,72,122,585).  visible(lb2gt_6,585). happensAt(inactive(lb2gt_6),585).
orientation(lb2gt_4,130,586). coords(lb2gt_4,141,37,586).  visible(lb2gt_4,586). happensAt(walking(lb2gt_4),586). orientation(lb2gt_6,151,586). coords(lb2gt_6,72,122,586).  visible(lb2gt_6,586). happensAt(inactive(lb2gt_6),586).
orientation(lb2gt_4,130,587). coords(lb2gt_4,142,36,587).  visible(lb2gt_4,587). happensAt(walking(lb2gt_4),587). orientation(lb2gt_6,151,587). coords(lb2gt_6,72,122,587).  visible(lb2gt_6,587). happensAt(inactive(lb2gt_6),587).
orientation(lb2gt_4,130,588). coords(lb2gt_4,143,36,588).  visible(lb2gt_4,588). happensAt(walking(lb2gt_4),588). orientation(lb2gt_6,151,588). coords(lb2gt_6,72,122,588).  visible(lb2gt_6,588). happensAt(inactive(lb2gt_6),588).
orientation(lb2gt_4,130,589). coords(lb2gt_4,143,36,589).  visible(lb2gt_4,589). happensAt(walking(lb2gt_4),589). orientation(lb2gt_6,151,589). coords(lb2gt_6,72,122,589).  visible(lb2gt_6,589). happensAt(inactive(lb2gt_6),589).
orientation(lb2gt_4,130,590). coords(lb2gt_4,143,35,590).  visible(lb2gt_4,590). happensAt(walking(lb2gt_4),590). orientation(lb2gt_6,151,590). coords(lb2gt_6,72,122,590).  visible(lb2gt_6,590). happensAt(inactive(lb2gt_6),590).
orientation(lb2gt_4,130,591). coords(lb2gt_4,145,35,591).  visible(lb2gt_4,591). happensAt(walking(lb2gt_4),591). orientation(lb2gt_6,151,591). coords(lb2gt_6,72,122,591).  visible(lb2gt_6,591). happensAt(inactive(lb2gt_6),591).
orientation(lb2gt_4,130,592). coords(lb2gt_4,145,35,592).  visible(lb2gt_4,592). happensAt(walking(lb2gt_4),592). orientation(lb2gt_6,151,592). coords(lb2gt_6,72,122,592).  visible(lb2gt_6,592). happensAt(inactive(lb2gt_6),592).
orientation(lb2gt_4,130,593). coords(lb2gt_4,145,34,593).  visible(lb2gt_4,593). happensAt(walking(lb2gt_4),593). orientation(lb2gt_6,151,593). coords(lb2gt_6,72,122,593).  visible(lb2gt_6,593). happensAt(inactive(lb2gt_6),593).
orientation(lb2gt_4,130,594). coords(lb2gt_4,146,33,594).  visible(lb2gt_4,594). happensAt(walking(lb2gt_4),594). orientation(lb2gt_6,151,594). coords(lb2gt_6,72,122,594).  visible(lb2gt_6,594). happensAt(inactive(lb2gt_6),594).
orientation(lb2gt_4,130,595). coords(lb2gt_4,146,33,595).  visible(lb2gt_4,595). happensAt(walking(lb2gt_4),595). orientation(lb2gt_6,151,595). coords(lb2gt_6,72,122,595).  visible(lb2gt_6,595). happensAt(inactive(lb2gt_6),595).
orientation(lb2gt_4,130,596). coords(lb2gt_4,147,32,596).  visible(lb2gt_4,596). happensAt(walking(lb2gt_4),596). orientation(lb2gt_6,151,596). coords(lb2gt_6,72,122,596).  visible(lb2gt_6,596). happensAt(inactive(lb2gt_6),596).
orientation(lb2gt_4,130,597). coords(lb2gt_4,147,31,597).  visible(lb2gt_4,597). happensAt(walking(lb2gt_4),597). orientation(lb2gt_6,151,597). coords(lb2gt_6,72,122,597).  visible(lb2gt_6,597). happensAt(inactive(lb2gt_6),597).
orientation(lb2gt_4,130,598). coords(lb2gt_4,147,30,598).  visible(lb2gt_4,598). happensAt(walking(lb2gt_4),598). orientation(lb2gt_6,151,598). coords(lb2gt_6,72,122,598).  visible(lb2gt_6,598). happensAt(inactive(lb2gt_6),598).
orientation(lb2gt_4,130,599). coords(lb2gt_4,147,30,599).  visible(lb2gt_4,599). happensAt(walking(lb2gt_4),599). orientation(lb2gt_6,151,599). coords(lb2gt_6,72,122,599).  visible(lb2gt_6,599). happensAt(inactive(lb2gt_6),599).
#show holdsAt/2.