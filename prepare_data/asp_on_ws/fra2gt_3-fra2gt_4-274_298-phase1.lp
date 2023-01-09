
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

	orientation(fra2gt_3,156,274). coords(fra2gt_3,75,129,274).  visible(fra2gt_3,274). happensAt(walking(fra2gt_3),274). orientation(fra2gt_4,155,274). coords(fra2gt_4,135,136,274).  visible(fra2gt_4,274). happensAt(walking(fra2gt_4),274).
orientation(fra2gt_3,156,275). coords(fra2gt_3,75,129,275).  visible(fra2gt_3,275). happensAt(walking(fra2gt_3),275). orientation(fra2gt_4,155,275). coords(fra2gt_4,134,132,275).  visible(fra2gt_4,275). happensAt(walking(fra2gt_4),275).
orientation(fra2gt_3,156,276). coords(fra2gt_3,76,128,276).  visible(fra2gt_3,276). happensAt(walking(fra2gt_3),276). orientation(fra2gt_4,155,276). coords(fra2gt_4,133,130,276).  visible(fra2gt_4,276). happensAt(walking(fra2gt_4),276).
orientation(fra2gt_3,156,277). coords(fra2gt_3,77,128,277).  visible(fra2gt_3,277). happensAt(walking(fra2gt_3),277). orientation(fra2gt_4,155,277). coords(fra2gt_4,132,129,277).  visible(fra2gt_4,277). happensAt(walking(fra2gt_4),277).
orientation(fra2gt_3,156,278). coords(fra2gt_3,79,128,278).  visible(fra2gt_3,278). happensAt(walking(fra2gt_3),278). orientation(fra2gt_4,155,278). coords(fra2gt_4,131,129,278).  visible(fra2gt_4,278). happensAt(walking(fra2gt_4),278).
orientation(fra2gt_3,156,279). coords(fra2gt_3,80,128,279).  visible(fra2gt_3,279). happensAt(walking(fra2gt_3),279). orientation(fra2gt_4,155,279). coords(fra2gt_4,130,127,279).  visible(fra2gt_4,279). happensAt(walking(fra2gt_4),279).
orientation(fra2gt_3,156,280). coords(fra2gt_3,80,127,280).  visible(fra2gt_3,280). happensAt(walking(fra2gt_3),280). orientation(fra2gt_4,155,280). coords(fra2gt_4,128,128,280).  visible(fra2gt_4,280). happensAt(walking(fra2gt_4),280).
orientation(fra2gt_3,156,281). coords(fra2gt_3,80,127,281).  visible(fra2gt_3,281). happensAt(walking(fra2gt_3),281). orientation(fra2gt_4,155,281). coords(fra2gt_4,125,127,281).  visible(fra2gt_4,281). happensAt(walking(fra2gt_4),281).
orientation(fra2gt_3,156,282). coords(fra2gt_3,80,126,282).  visible(fra2gt_3,282). happensAt(walking(fra2gt_3),282). orientation(fra2gt_4,155,282). coords(fra2gt_4,123,126,282).  visible(fra2gt_4,282). happensAt(walking(fra2gt_4),282).
orientation(fra2gt_3,156,283). coords(fra2gt_3,81,126,283).  visible(fra2gt_3,283). happensAt(walking(fra2gt_3),283). orientation(fra2gt_4,155,283). coords(fra2gt_4,122,126,283).  visible(fra2gt_4,283). happensAt(walking(fra2gt_4),283).
orientation(fra2gt_3,153,284). coords(fra2gt_3,81,126,284).  visible(fra2gt_3,284). happensAt(walking(fra2gt_3),284). orientation(fra2gt_4,155,284). coords(fra2gt_4,121,127,284).  visible(fra2gt_4,284). happensAt(walking(fra2gt_4),284).
orientation(fra2gt_3,153,285). coords(fra2gt_3,81,124,285).  visible(fra2gt_3,285). happensAt(walking(fra2gt_3),285). orientation(fra2gt_4,155,285). coords(fra2gt_4,121,128,285).  visible(fra2gt_4,285). happensAt(walking(fra2gt_4),285).
orientation(fra2gt_3,153,286). coords(fra2gt_3,81,124,286).  visible(fra2gt_3,286). happensAt(walking(fra2gt_3),286). orientation(fra2gt_4,155,286). coords(fra2gt_4,120,129,286).  visible(fra2gt_4,286). happensAt(walking(fra2gt_4),286).
orientation(fra2gt_3,153,287). coords(fra2gt_3,82,123,287).  visible(fra2gt_3,287). happensAt(walking(fra2gt_3),287). orientation(fra2gt_4,155,287). coords(fra2gt_4,120,130,287).  visible(fra2gt_4,287). happensAt(walking(fra2gt_4),287).
orientation(fra2gt_3,153,288). coords(fra2gt_3,83,122,288).  visible(fra2gt_3,288). happensAt(walking(fra2gt_3),288). orientation(fra2gt_4,155,288). coords(fra2gt_4,119,132,288).  visible(fra2gt_4,288). happensAt(walking(fra2gt_4),288).
orientation(fra2gt_3,150,289). coords(fra2gt_3,84,122,289).  visible(fra2gt_3,289). happensAt(walking(fra2gt_3),289). orientation(fra2gt_4,155,289). coords(fra2gt_4,117,131,289).  visible(fra2gt_4,289). happensAt(walking(fra2gt_4),289).
orientation(fra2gt_3,150,290). coords(fra2gt_3,84,122,290).  visible(fra2gt_3,290). happensAt(walking(fra2gt_3),290). orientation(fra2gt_4,155,290). coords(fra2gt_4,117,130,290).  visible(fra2gt_4,290). happensAt(walking(fra2gt_4),290).
orientation(fra2gt_3,150,291). coords(fra2gt_3,85,121,291).  visible(fra2gt_3,291). happensAt(walking(fra2gt_3),291). orientation(fra2gt_4,155,291). coords(fra2gt_4,116,129,291).  visible(fra2gt_4,291). happensAt(walking(fra2gt_4),291).
orientation(fra2gt_3,150,292). coords(fra2gt_3,85,121,292).  visible(fra2gt_3,292). happensAt(walking(fra2gt_3),292). orientation(fra2gt_4,155,292). coords(fra2gt_4,115,128,292).  visible(fra2gt_4,292). happensAt(walking(fra2gt_4),292).
orientation(fra2gt_3,150,293). coords(fra2gt_3,84,121,293).  visible(fra2gt_3,293). happensAt(walking(fra2gt_3),293). orientation(fra2gt_4,155,293). coords(fra2gt_4,113,128,293).  visible(fra2gt_4,293). happensAt(walking(fra2gt_4),293).
orientation(fra2gt_3,150,294). coords(fra2gt_3,84,120,294).  visible(fra2gt_3,294). happensAt(walking(fra2gt_3),294). orientation(fra2gt_4,155,294). coords(fra2gt_4,112,128,294).  visible(fra2gt_4,294). happensAt(walking(fra2gt_4),294).
orientation(fra2gt_3,150,295). coords(fra2gt_3,84,120,295).  visible(fra2gt_3,295). happensAt(walking(fra2gt_3),295). orientation(fra2gt_4,155,295). coords(fra2gt_4,112,127,295).  visible(fra2gt_4,295). happensAt(walking(fra2gt_4),295).
orientation(fra2gt_3,150,296). coords(fra2gt_3,84,119,296).  visible(fra2gt_3,296). happensAt(walking(fra2gt_3),296). orientation(fra2gt_4,155,296). coords(fra2gt_4,111,125,296).  visible(fra2gt_4,296). happensAt(walking(fra2gt_4),296).
orientation(fra2gt_3,150,297). coords(fra2gt_3,83,118,297).  visible(fra2gt_3,297). happensAt(walking(fra2gt_3),297). orientation(fra2gt_4,155,297). coords(fra2gt_4,109,123,297).  visible(fra2gt_4,297). happensAt(walking(fra2gt_4),297).
orientation(fra2gt_3,150,298). coords(fra2gt_3,85,117,298).  visible(fra2gt_3,298). happensAt(walking(fra2gt_3),298). orientation(fra2gt_4,155,298). coords(fra2gt_4,108,121,298).  visible(fra2gt_4,298). happensAt(walking(fra2gt_4),298).
#show holdsAt/2.