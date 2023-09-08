
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

	orientation(wk2gt_1,125,276). coords(wk2gt_1,115,53,276).  visible(wk2gt_1,276). happensAt(inactive(wk2gt_1),276). orientation(wk2gt_2,141,276). coords(wk2gt_2,98,63,276).  visible(wk2gt_2,276). happensAt(inactive(wk2gt_2),276).
orientation(wk2gt_1,125,277). coords(wk2gt_1,115,53,277).  visible(wk2gt_1,277). happensAt(inactive(wk2gt_1),277). orientation(wk2gt_2,141,277). coords(wk2gt_2,98,63,277).  visible(wk2gt_2,277). happensAt(inactive(wk2gt_2),277).
orientation(wk2gt_1,125,278). coords(wk2gt_1,115,53,278).  visible(wk2gt_1,278). happensAt(inactive(wk2gt_1),278). orientation(wk2gt_2,141,278). coords(wk2gt_2,98,63,278).  visible(wk2gt_2,278). happensAt(inactive(wk2gt_2),278).
orientation(wk2gt_1,125,279). coords(wk2gt_1,115,53,279).  visible(wk2gt_1,279). happensAt(inactive(wk2gt_1),279). orientation(wk2gt_2,141,279). coords(wk2gt_2,98,63,279).  visible(wk2gt_2,279). happensAt(inactive(wk2gt_2),279).
orientation(wk2gt_1,125,280). coords(wk2gt_1,115,53,280).  visible(wk2gt_1,280). happensAt(inactive(wk2gt_1),280). orientation(wk2gt_2,141,280). coords(wk2gt_2,98,63,280).  visible(wk2gt_2,280). happensAt(inactive(wk2gt_2),280).
orientation(wk2gt_1,125,281). coords(wk2gt_1,115,53,281).  visible(wk2gt_1,281). happensAt(inactive(wk2gt_1),281). orientation(wk2gt_2,141,281). coords(wk2gt_2,98,63,281).  visible(wk2gt_2,281). happensAt(inactive(wk2gt_2),281).
orientation(wk2gt_1,125,282). coords(wk2gt_1,115,53,282).  visible(wk2gt_1,282). happensAt(inactive(wk2gt_1),282). orientation(wk2gt_2,141,282). coords(wk2gt_2,98,63,282).  visible(wk2gt_2,282). happensAt(inactive(wk2gt_2),282).
orientation(wk2gt_1,125,283). coords(wk2gt_1,115,53,283).  visible(wk2gt_1,283). happensAt(inactive(wk2gt_1),283). orientation(wk2gt_2,141,283). coords(wk2gt_2,98,63,283).  visible(wk2gt_2,283). happensAt(inactive(wk2gt_2),283).
orientation(wk2gt_1,125,284). coords(wk2gt_1,115,53,284).  visible(wk2gt_1,284). happensAt(inactive(wk2gt_1),284). orientation(wk2gt_2,141,284). coords(wk2gt_2,98,63,284).  visible(wk2gt_2,284). happensAt(inactive(wk2gt_2),284).
orientation(wk2gt_1,125,285). coords(wk2gt_1,115,53,285).  visible(wk2gt_1,285). happensAt(inactive(wk2gt_1),285). orientation(wk2gt_2,141,285). coords(wk2gt_2,98,63,285).  visible(wk2gt_2,285). happensAt(inactive(wk2gt_2),285).
orientation(wk2gt_1,125,286). coords(wk2gt_1,115,53,286).  visible(wk2gt_1,286). happensAt(inactive(wk2gt_1),286). orientation(wk2gt_2,141,286). coords(wk2gt_2,98,63,286).  visible(wk2gt_2,286). happensAt(inactive(wk2gt_2),286).
orientation(wk2gt_1,125,287). coords(wk2gt_1,115,53,287).  visible(wk2gt_1,287). happensAt(inactive(wk2gt_1),287). orientation(wk2gt_2,141,287). coords(wk2gt_2,98,63,287).  visible(wk2gt_2,287). happensAt(inactive(wk2gt_2),287).
orientation(wk2gt_1,125,288). coords(wk2gt_1,115,53,288).  visible(wk2gt_1,288). happensAt(inactive(wk2gt_1),288). orientation(wk2gt_2,141,288). coords(wk2gt_2,98,63,288).  visible(wk2gt_2,288). happensAt(inactive(wk2gt_2),288).
orientation(wk2gt_1,125,289). coords(wk2gt_1,115,53,289).  visible(wk2gt_1,289). happensAt(inactive(wk2gt_1),289). orientation(wk2gt_2,141,289). coords(wk2gt_2,98,63,289).  visible(wk2gt_2,289). happensAt(inactive(wk2gt_2),289).
orientation(wk2gt_1,125,290). coords(wk2gt_1,115,53,290).  visible(wk2gt_1,290). happensAt(inactive(wk2gt_1),290). orientation(wk2gt_2,141,290). coords(wk2gt_2,98,63,290).  visible(wk2gt_2,290). happensAt(inactive(wk2gt_2),290).
orientation(wk2gt_1,125,291). coords(wk2gt_1,115,53,291).  visible(wk2gt_1,291). happensAt(inactive(wk2gt_1),291). orientation(wk2gt_2,141,291). coords(wk2gt_2,98,63,291).  visible(wk2gt_2,291). happensAt(inactive(wk2gt_2),291).
orientation(wk2gt_1,125,292). coords(wk2gt_1,115,53,292).  visible(wk2gt_1,292). happensAt(inactive(wk2gt_1),292). orientation(wk2gt_2,141,292). coords(wk2gt_2,98,63,292).  visible(wk2gt_2,292). happensAt(inactive(wk2gt_2),292).
orientation(wk2gt_1,125,293). coords(wk2gt_1,115,53,293).  visible(wk2gt_1,293). happensAt(inactive(wk2gt_1),293). orientation(wk2gt_2,141,293). coords(wk2gt_2,98,63,293).  visible(wk2gt_2,293). happensAt(inactive(wk2gt_2),293).
orientation(wk2gt_1,125,294). coords(wk2gt_1,115,53,294).  visible(wk2gt_1,294). happensAt(inactive(wk2gt_1),294). orientation(wk2gt_2,141,294). coords(wk2gt_2,98,63,294).  visible(wk2gt_2,294). happensAt(inactive(wk2gt_2),294).
orientation(wk2gt_1,125,295). coords(wk2gt_1,115,53,295).  visible(wk2gt_1,295). happensAt(inactive(wk2gt_1),295). orientation(wk2gt_2,141,295). coords(wk2gt_2,98,63,295).  visible(wk2gt_2,295). happensAt(inactive(wk2gt_2),295).
orientation(wk2gt_1,125,296). coords(wk2gt_1,115,53,296).  visible(wk2gt_1,296). happensAt(inactive(wk2gt_1),296). orientation(wk2gt_2,141,296). coords(wk2gt_2,98,63,296).  visible(wk2gt_2,296). happensAt(inactive(wk2gt_2),296).
orientation(wk2gt_1,125,297). coords(wk2gt_1,115,53,297).  visible(wk2gt_1,297). happensAt(inactive(wk2gt_1),297). orientation(wk2gt_2,141,297). coords(wk2gt_2,98,63,297).  visible(wk2gt_2,297). happensAt(inactive(wk2gt_2),297).
orientation(wk2gt_1,125,298). coords(wk2gt_1,115,53,298).  visible(wk2gt_1,298). happensAt(inactive(wk2gt_1),298). orientation(wk2gt_2,141,298). coords(wk2gt_2,98,63,298).  visible(wk2gt_2,298). happensAt(inactive(wk2gt_2),298).
orientation(wk2gt_1,125,299). coords(wk2gt_1,115,53,299).  visible(wk2gt_1,299). happensAt(inactive(wk2gt_1),299). orientation(wk2gt_2,141,299). coords(wk2gt_2,98,63,299).  visible(wk2gt_2,299). happensAt(inactive(wk2gt_2),299).
orientation(wk2gt_1,125,300). coords(wk2gt_1,115,53,300).  visible(wk2gt_1,300). happensAt(inactive(wk2gt_1),300). orientation(wk2gt_2,141,300). coords(wk2gt_2,98,63,300).  visible(wk2gt_2,300). happensAt(inactive(wk2gt_2),300).
#show holdsAt/2.