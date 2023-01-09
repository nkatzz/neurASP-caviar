
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

	orientation(fomdgt3_2,130,576). coords(fomdgt3_2,247,93,576).  visible(fomdgt3_2,576). happensAt(walking(fomdgt3_2),576). orientation(fomdgt3_7,171,576). coords(fomdgt3_7,222,260,576).  visible(fomdgt3_7,576). happensAt(walking(fomdgt3_7),576).
orientation(fomdgt3_2,130,577). coords(fomdgt3_2,247,94,577).  visible(fomdgt3_2,577). happensAt(walking(fomdgt3_2),577). orientation(fomdgt3_7,171,577). coords(fomdgt3_7,225,261,577).  visible(fomdgt3_7,577). happensAt(walking(fomdgt3_7),577).
orientation(fomdgt3_2,130,578). coords(fomdgt3_2,248,94,578).  visible(fomdgt3_2,578). happensAt(walking(fomdgt3_2),578). orientation(fomdgt3_7,171,578). coords(fomdgt3_7,225,262,578).  visible(fomdgt3_7,578). happensAt(walking(fomdgt3_7),578).
orientation(fomdgt3_2,130,579). coords(fomdgt3_2,248,95,579).  visible(fomdgt3_2,579). happensAt(walking(fomdgt3_2),579). orientation(fomdgt3_7,171,579). coords(fomdgt3_7,227,263,579).  visible(fomdgt3_7,579). happensAt(walking(fomdgt3_7),579).
orientation(fomdgt3_2,130,580). coords(fomdgt3_2,249,97,580).  visible(fomdgt3_2,580). happensAt(walking(fomdgt3_2),580). orientation(fomdgt3_7,171,580). coords(fomdgt3_7,227,263,580).  visible(fomdgt3_7,580). happensAt(walking(fomdgt3_7),580).
orientation(fomdgt3_2,130,581). coords(fomdgt3_2,249,98,581).  visible(fomdgt3_2,581). happensAt(walking(fomdgt3_2),581). orientation(fomdgt3_7,20,581). coords(fomdgt3_7,227,264,581).  visible(fomdgt3_7,581). happensAt(walking(fomdgt3_7),581).
orientation(fomdgt3_2,130,582). coords(fomdgt3_2,249,99,582).  visible(fomdgt3_2,582). happensAt(walking(fomdgt3_2),582). orientation(fomdgt3_7,20,582). coords(fomdgt3_7,227,264,582).  visible(fomdgt3_7,582). happensAt(walking(fomdgt3_7),582).
orientation(fomdgt3_2,130,583). coords(fomdgt3_2,248,101,583).  visible(fomdgt3_2,583). happensAt(walking(fomdgt3_2),583). orientation(fomdgt3_7,20,583). coords(fomdgt3_7,227,266,583).  visible(fomdgt3_7,583). happensAt(walking(fomdgt3_7),583).
orientation(fomdgt3_2,130,584). coords(fomdgt3_2,249,101,584).  visible(fomdgt3_2,584). happensAt(walking(fomdgt3_2),584). orientation(fomdgt3_7,20,584). coords(fomdgt3_7,229,268,584).  visible(fomdgt3_7,584). happensAt(walking(fomdgt3_7),584).
orientation(fomdgt3_2,130,585). coords(fomdgt3_2,249,101,585).  visible(fomdgt3_2,585). happensAt(walking(fomdgt3_2),585). orientation(fomdgt3_7,30,585). coords(fomdgt3_7,229,270,585).  visible(fomdgt3_7,585). happensAt(walking(fomdgt3_7),585).
orientation(fomdgt3_2,130,586). coords(fomdgt3_2,249,102,586).  visible(fomdgt3_2,586). happensAt(walking(fomdgt3_2),586). orientation(fomdgt3_7,30,586). coords(fomdgt3_7,229,272,586).  visible(fomdgt3_7,586). happensAt(walking(fomdgt3_7),586).
orientation(fomdgt3_2,130,587). coords(fomdgt3_2,250,103,587).  visible(fomdgt3_2,587). happensAt(walking(fomdgt3_2),587). orientation(fomdgt3_7,30,587). coords(fomdgt3_7,229,274,587).  visible(fomdgt3_7,587). happensAt(walking(fomdgt3_7),587).
orientation(fomdgt3_2,130,588). coords(fomdgt3_2,250,103,588).  visible(fomdgt3_2,588). happensAt(walking(fomdgt3_2),588). orientation(fomdgt3_7,30,588). coords(fomdgt3_7,231,274,588).  visible(fomdgt3_7,588). happensAt(walking(fomdgt3_7),588).
orientation(fomdgt3_2,130,589). coords(fomdgt3_2,250,104,589).  visible(fomdgt3_2,589). happensAt(walking(fomdgt3_2),589). orientation(fomdgt3_7,30,589). coords(fomdgt3_7,231,276,589).  visible(fomdgt3_7,589). happensAt(walking(fomdgt3_7),589).
orientation(fomdgt3_2,130,590). coords(fomdgt3_2,250,105,590).  visible(fomdgt3_2,590). happensAt(walking(fomdgt3_2),590). orientation(fomdgt3_7,30,590). coords(fomdgt3_7,231,276,590).  visible(fomdgt3_7,590). happensAt(walking(fomdgt3_7),590).
orientation(fomdgt3_2,130,591). coords(fomdgt3_2,250,106,591).  visible(fomdgt3_2,591). happensAt(walking(fomdgt3_2),591). orientation(fomdgt3_7,30,591). coords(fomdgt3_7,231,276,591).  visible(fomdgt3_7,591). happensAt(walking(fomdgt3_7),591).
orientation(fomdgt3_2,130,592). coords(fomdgt3_2,251,107,592).  visible(fomdgt3_2,592). happensAt(walking(fomdgt3_2),592). orientation(fomdgt3_7,30,592). coords(fomdgt3_7,231,276,592).  visible(fomdgt3_7,592). happensAt(walking(fomdgt3_7),592).
orientation(fomdgt3_2,130,593). coords(fomdgt3_2,250,107,593).  visible(fomdgt3_2,593). happensAt(walking(fomdgt3_2),593). orientation(fomdgt3_7,30,593). coords(fomdgt3_7,231,276,593).  visible(fomdgt3_7,593). happensAt(walking(fomdgt3_7),593).
orientation(fomdgt3_2,130,594). coords(fomdgt3_2,251,108,594).  visible(fomdgt3_2,594). happensAt(walking(fomdgt3_2),594). orientation(fomdgt3_7,30,594). coords(fomdgt3_7,234,276,594).  visible(fomdgt3_7,594). happensAt(walking(fomdgt3_7),594).
orientation(fomdgt3_2,130,595). coords(fomdgt3_2,252,108,595).  visible(fomdgt3_2,595). happensAt(walking(fomdgt3_2),595). orientation(fomdgt3_7,70,595). coords(fomdgt3_7,244,276,595).  visible(fomdgt3_7,595). happensAt(walking(fomdgt3_7),595).
orientation(fomdgt3_2,130,596). coords(fomdgt3_2,252,109,596).  visible(fomdgt3_2,596). happensAt(walking(fomdgt3_2),596). orientation(fomdgt3_7,70,596). coords(fomdgt3_7,244,277,596).  visible(fomdgt3_7,596). happensAt(walking(fomdgt3_7),596).
orientation(fomdgt3_2,130,597). coords(fomdgt3_2,252,109,597).  visible(fomdgt3_2,597). happensAt(walking(fomdgt3_2),597). orientation(fomdgt3_7,70,597). coords(fomdgt3_7,244,278,597).  visible(fomdgt3_7,597). happensAt(walking(fomdgt3_7),597).
orientation(fomdgt3_2,130,598). coords(fomdgt3_2,252,109,598).  visible(fomdgt3_2,598). happensAt(walking(fomdgt3_2),598). orientation(fomdgt3_7,70,598). coords(fomdgt3_7,244,279,598).  visible(fomdgt3_7,598). happensAt(walking(fomdgt3_7),598).
orientation(fomdgt3_2,130,599). coords(fomdgt3_2,252,109,599).  visible(fomdgt3_2,599). happensAt(walking(fomdgt3_2),599). orientation(fomdgt3_7,70,599). coords(fomdgt3_7,244,280,599).  visible(fomdgt3_7,599). happensAt(walking(fomdgt3_7),599).
orientation(fomdgt3_2,130,600). coords(fomdgt3_2,252,109,600).  visible(fomdgt3_2,600). happensAt(walking(fomdgt3_2),600). orientation(fomdgt3_7,70,600). coords(fomdgt3_7,244,280,600).  happensAt(disappear(fomdgt3_7),600). happensAt(walking(fomdgt3_7),600).
#show holdsAt/2.