
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

	orientation(mc1gt_1,170,276). coords(mc1gt_1,217,204,276).  visible(mc1gt_1,276). happensAt(walking(mc1gt_1),276). orientation(mc1gt_3,135,276). coords(mc1gt_3,235,128,276).  visible(mc1gt_3,276). happensAt(walking(mc1gt_3),276).
orientation(mc1gt_1,169,277). coords(mc1gt_1,218,208,277).  visible(mc1gt_1,277). happensAt(walking(mc1gt_1),277). orientation(mc1gt_3,137,277). coords(mc1gt_3,237,129,277).  visible(mc1gt_3,277). happensAt(walking(mc1gt_3),277).
orientation(mc1gt_1,175,278). coords(mc1gt_1,218,210,278).  visible(mc1gt_1,278). happensAt(walking(mc1gt_1),278). orientation(mc1gt_3,135,278). coords(mc1gt_3,241,130,278).  visible(mc1gt_3,278). happensAt(walking(mc1gt_3),278).
orientation(mc1gt_1,175,279). coords(mc1gt_1,219,211,279).  visible(mc1gt_1,279). happensAt(walking(mc1gt_1),279). orientation(mc1gt_3,137,279). coords(mc1gt_3,245,130,279).  visible(mc1gt_3,279). happensAt(walking(mc1gt_3),279).
orientation(mc1gt_1,174,280). coords(mc1gt_1,219,211,280).  visible(mc1gt_1,280). happensAt(walking(mc1gt_1),280). orientation(mc1gt_3,137,280). coords(mc1gt_3,249,133,280).  visible(mc1gt_3,280). happensAt(walking(mc1gt_3),280).
orientation(mc1gt_1,178,281). coords(mc1gt_1,220,211,281).  visible(mc1gt_1,281). happensAt(walking(mc1gt_1),281). orientation(mc1gt_3,137,281). coords(mc1gt_3,252,136,281).  visible(mc1gt_3,281). happensAt(walking(mc1gt_3),281).
orientation(mc1gt_1,1,282). coords(mc1gt_1,221,211,282).  visible(mc1gt_1,282). happensAt(walking(mc1gt_1),282). orientation(mc1gt_3,136,282). coords(mc1gt_3,255,139,282).  visible(mc1gt_3,282). happensAt(walking(mc1gt_3),282).
orientation(mc1gt_1,6,283). coords(mc1gt_1,221,212,283).  visible(mc1gt_1,283). happensAt(walking(mc1gt_1),283). orientation(mc1gt_3,137,283). coords(mc1gt_3,257,140,283).  visible(mc1gt_3,283). happensAt(walking(mc1gt_3),283).
orientation(mc1gt_1,4,284). coords(mc1gt_1,222,213,284).  visible(mc1gt_1,284). happensAt(walking(mc1gt_1),284). orientation(mc1gt_3,135,284). coords(mc1gt_3,259,142,284).  visible(mc1gt_3,284). happensAt(walking(mc1gt_3),284).
orientation(mc1gt_1,4,285). coords(mc1gt_1,222,215,285).  visible(mc1gt_1,285). happensAt(walking(mc1gt_1),285). orientation(mc1gt_3,135,285). coords(mc1gt_3,260,143,285).  visible(mc1gt_3,285). happensAt(walking(mc1gt_3),285).
orientation(mc1gt_1,7,286). coords(mc1gt_1,222,218,286).  visible(mc1gt_1,286). happensAt(walking(mc1gt_1),286). orientation(mc1gt_3,135,286). coords(mc1gt_3,261,144,286).  visible(mc1gt_3,286). happensAt(walking(mc1gt_3),286).
orientation(mc1gt_1,3,287). coords(mc1gt_1,222,221,287).  visible(mc1gt_1,287). happensAt(walking(mc1gt_1),287). orientation(mc1gt_3,135,287). coords(mc1gt_3,262,145,287).  visible(mc1gt_3,287). happensAt(walking(mc1gt_3),287).
orientation(mc1gt_1,1,288). coords(mc1gt_1,222,225,288).  visible(mc1gt_1,288). happensAt(walking(mc1gt_1),288). orientation(mc1gt_3,135,288). coords(mc1gt_3,263,146,288).  visible(mc1gt_3,288). happensAt(walking(mc1gt_3),288).
orientation(mc1gt_1,3,289). coords(mc1gt_1,222,228,289).  visible(mc1gt_1,289). happensAt(walking(mc1gt_1),289). orientation(mc1gt_3,135,289). coords(mc1gt_3,264,148,289).  visible(mc1gt_3,289). happensAt(walking(mc1gt_3),289).
orientation(mc1gt_1,0,290). coords(mc1gt_1,222,230,290).  visible(mc1gt_1,290). happensAt(walking(mc1gt_1),290). orientation(mc1gt_3,135,290). coords(mc1gt_3,264,150,290).  visible(mc1gt_3,290). happensAt(walking(mc1gt_3),290).
orientation(mc1gt_1,171,291). coords(mc1gt_1,222,232,291).  visible(mc1gt_1,291). happensAt(walking(mc1gt_1),291). orientation(mc1gt_3,135,291). coords(mc1gt_3,265,153,291).  visible(mc1gt_3,291). happensAt(walking(mc1gt_3),291).
orientation(mc1gt_1,172,292). coords(mc1gt_1,222,235,292).  visible(mc1gt_1,292). happensAt(walking(mc1gt_1),292). orientation(mc1gt_3,135,292). coords(mc1gt_3,269,158,292).  visible(mc1gt_3,292). happensAt(walking(mc1gt_3),292).
orientation(mc1gt_1,174,293). coords(mc1gt_1,222,237,293).  visible(mc1gt_1,293). happensAt(walking(mc1gt_1),293). orientation(mc1gt_3,130,293). coords(mc1gt_3,271,162,293).  visible(mc1gt_3,293). happensAt(walking(mc1gt_3),293).
orientation(mc1gt_1,0,294). coords(mc1gt_1,222,239,294).  visible(mc1gt_1,294). happensAt(walking(mc1gt_1),294). orientation(mc1gt_3,125,294). coords(mc1gt_3,274,166,294).  visible(mc1gt_3,294). happensAt(walking(mc1gt_3),294).
orientation(mc1gt_1,3,295). coords(mc1gt_1,222,239,295).  visible(mc1gt_1,295). happensAt(walking(mc1gt_1),295). orientation(mc1gt_3,123,295). coords(mc1gt_3,276,169,295).  visible(mc1gt_3,295). happensAt(walking(mc1gt_3),295).
orientation(mc1gt_1,178,296). coords(mc1gt_1,222,240,296).  visible(mc1gt_1,296). happensAt(walking(mc1gt_1),296). orientation(mc1gt_3,124,296). coords(mc1gt_3,278,172,296).  visible(mc1gt_3,296). happensAt(walking(mc1gt_3),296).
orientation(mc1gt_1,3,297). coords(mc1gt_1,222,240,297).  visible(mc1gt_1,297). happensAt(walking(mc1gt_1),297). orientation(mc1gt_3,124,297). coords(mc1gt_3,280,174,297).  visible(mc1gt_3,297). happensAt(walking(mc1gt_3),297).
orientation(mc1gt_1,6,298). coords(mc1gt_1,222,241,298).  visible(mc1gt_1,298). happensAt(walking(mc1gt_1),298). orientation(mc1gt_3,131,298). coords(mc1gt_3,281,176,298).  visible(mc1gt_3,298). happensAt(walking(mc1gt_3),298).
orientation(mc1gt_1,0,299). coords(mc1gt_1,222,243,299).  visible(mc1gt_1,299). happensAt(walking(mc1gt_1),299). orientation(mc1gt_3,134,299). coords(mc1gt_3,283,176,299).  visible(mc1gt_3,299). happensAt(walking(mc1gt_3),299).
orientation(mc1gt_1,1,300). coords(mc1gt_1,222,245,300).  visible(mc1gt_1,300). happensAt(walking(mc1gt_1),300). orientation(mc1gt_3,135,300). coords(mc1gt_3,283,177,300).  visible(mc1gt_3,300). happensAt(walking(mc1gt_3),300).
#show holdsAt/2.