
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

	orientation(mws1gt_5,161,276). coords(mws1gt_5,39,158,276).  visible(mws1gt_5,276). happensAt(walking(mws1gt_5),276). orientation(mws1gt_6,146,276). coords(mws1gt_6,40,136,276).  visible(mws1gt_6,276). happensAt(walking(mws1gt_6),276).
orientation(mws1gt_5,161,277). coords(mws1gt_5,39,159,277).  visible(mws1gt_5,277). happensAt(walking(mws1gt_5),277). orientation(mws1gt_6,146,277). coords(mws1gt_6,40,137,277).  visible(mws1gt_6,277). happensAt(walking(mws1gt_6),277).
orientation(mws1gt_5,161,278). coords(mws1gt_5,38,160,278).  visible(mws1gt_5,278). happensAt(walking(mws1gt_5),278). orientation(mws1gt_6,146,278). coords(mws1gt_6,40,138,278).  visible(mws1gt_6,278). happensAt(walking(mws1gt_6),278).
orientation(mws1gt_5,161,279). coords(mws1gt_5,38,162,279).  visible(mws1gt_5,279). happensAt(walking(mws1gt_5),279). orientation(mws1gt_6,146,279). coords(mws1gt_6,40,139,279).  visible(mws1gt_6,279). happensAt(walking(mws1gt_6),279).
orientation(mws1gt_5,161,280). coords(mws1gt_5,38,162,280).  visible(mws1gt_5,280). happensAt(walking(mws1gt_5),280). orientation(mws1gt_6,146,280). coords(mws1gt_6,40,140,280).  visible(mws1gt_6,280). happensAt(walking(mws1gt_6),280).
orientation(mws1gt_5,161,281). coords(mws1gt_5,38,163,281).  visible(mws1gt_5,281). happensAt(walking(mws1gt_5),281). orientation(mws1gt_6,146,281). coords(mws1gt_6,40,140,281).  visible(mws1gt_6,281). happensAt(walking(mws1gt_6),281).
orientation(mws1gt_5,161,282). coords(mws1gt_5,38,164,282).  visible(mws1gt_5,282). happensAt(walking(mws1gt_5),282). orientation(mws1gt_6,146,282). coords(mws1gt_6,39,140,282).  visible(mws1gt_6,282). happensAt(walking(mws1gt_6),282).
orientation(mws1gt_5,161,283). coords(mws1gt_5,38,164,283).  visible(mws1gt_5,283). happensAt(walking(mws1gt_5),283). orientation(mws1gt_6,146,283). coords(mws1gt_6,38,142,283).  visible(mws1gt_6,283). happensAt(walking(mws1gt_6),283).
orientation(mws1gt_5,161,284). coords(mws1gt_5,37,164,284).  visible(mws1gt_5,284). happensAt(walking(mws1gt_5),284). orientation(mws1gt_6,146,284). coords(mws1gt_6,38,143,284).  visible(mws1gt_6,284). happensAt(walking(mws1gt_6),284).
orientation(mws1gt_5,161,285). coords(mws1gt_5,37,165,285).  visible(mws1gt_5,285). happensAt(walking(mws1gt_5),285). orientation(mws1gt_6,146,285). coords(mws1gt_6,38,144,285).  visible(mws1gt_6,285). happensAt(walking(mws1gt_6),285).
orientation(mws1gt_5,161,286). coords(mws1gt_5,37,165,286).  visible(mws1gt_5,286). happensAt(walking(mws1gt_5),286). orientation(mws1gt_6,146,286). coords(mws1gt_6,38,145,286).  visible(mws1gt_6,286). happensAt(walking(mws1gt_6),286).
orientation(mws1gt_5,161,287). coords(mws1gt_5,37,166,287).  visible(mws1gt_5,287). happensAt(walking(mws1gt_5),287). orientation(mws1gt_6,146,287). coords(mws1gt_6,38,146,287).  visible(mws1gt_6,287). happensAt(walking(mws1gt_6),287).
orientation(mws1gt_5,161,288). coords(mws1gt_5,37,166,288).  visible(mws1gt_5,288). happensAt(walking(mws1gt_5),288). orientation(mws1gt_6,146,288). coords(mws1gt_6,38,146,288).  visible(mws1gt_6,288). happensAt(walking(mws1gt_6),288).
orientation(mws1gt_5,161,289). coords(mws1gt_5,37,167,289).  visible(mws1gt_5,289). happensAt(walking(mws1gt_5),289). orientation(mws1gt_6,146,289). coords(mws1gt_6,37,146,289).  visible(mws1gt_6,289). happensAt(walking(mws1gt_6),289).
orientation(mws1gt_5,161,290). coords(mws1gt_5,37,168,290).  visible(mws1gt_5,290). happensAt(walking(mws1gt_5),290). orientation(mws1gt_6,146,290). coords(mws1gt_6,37,147,290).  visible(mws1gt_6,290). happensAt(walking(mws1gt_6),290).
orientation(mws1gt_5,168,291). coords(mws1gt_5,37,169,291).  visible(mws1gt_5,291). happensAt(walking(mws1gt_5),291). orientation(mws1gt_6,146,291). coords(mws1gt_6,37,148,291).  visible(mws1gt_6,291). happensAt(walking(mws1gt_6),291).
orientation(mws1gt_5,168,292). coords(mws1gt_5,37,170,292).  visible(mws1gt_5,292). happensAt(walking(mws1gt_5),292). orientation(mws1gt_6,146,292). coords(mws1gt_6,36,150,292).  visible(mws1gt_6,292). happensAt(walking(mws1gt_6),292).
orientation(mws1gt_5,168,293). coords(mws1gt_5,37,171,293).  visible(mws1gt_5,293). happensAt(walking(mws1gt_5),293). orientation(mws1gt_6,146,293). coords(mws1gt_6,36,151,293).  visible(mws1gt_6,293). happensAt(walking(mws1gt_6),293).
orientation(mws1gt_5,168,294). coords(mws1gt_5,37,172,294).  visible(mws1gt_5,294). happensAt(walking(mws1gt_5),294). orientation(mws1gt_6,146,294). coords(mws1gt_6,36,152,294).  visible(mws1gt_6,294). happensAt(walking(mws1gt_6),294).
orientation(mws1gt_5,168,295). coords(mws1gt_5,37,174,295).  visible(mws1gt_5,295). happensAt(walking(mws1gt_5),295). orientation(mws1gt_6,146,295). coords(mws1gt_6,36,152,295).  visible(mws1gt_6,295). happensAt(walking(mws1gt_6),295).
orientation(mws1gt_5,168,296). coords(mws1gt_5,36,175,296).  visible(mws1gt_5,296). happensAt(walking(mws1gt_5),296). orientation(mws1gt_6,146,296). coords(mws1gt_6,36,153,296).  visible(mws1gt_6,296). happensAt(walking(mws1gt_6),296).
orientation(mws1gt_5,168,297). coords(mws1gt_5,36,176,297).  visible(mws1gt_5,297). happensAt(walking(mws1gt_5),297). orientation(mws1gt_6,146,297). coords(mws1gt_6,36,153,297).  visible(mws1gt_6,297). happensAt(walking(mws1gt_6),297).
orientation(mws1gt_5,168,298). coords(mws1gt_5,36,177,298).  visible(mws1gt_5,298). happensAt(walking(mws1gt_5),298). orientation(mws1gt_6,146,298). coords(mws1gt_6,36,154,298).  visible(mws1gt_6,298). happensAt(walking(mws1gt_6),298).
orientation(mws1gt_5,168,299). coords(mws1gt_5,36,178,299).  visible(mws1gt_5,299). happensAt(walking(mws1gt_5),299). orientation(mws1gt_6,146,299). coords(mws1gt_6,36,155,299).  visible(mws1gt_6,299). happensAt(walking(mws1gt_6),299).
orientation(mws1gt_5,168,300). coords(mws1gt_5,36,178,300).  visible(mws1gt_5,300). happensAt(walking(mws1gt_5),300). orientation(mws1gt_6,146,300). coords(mws1gt_6,36,155,300).  visible(mws1gt_6,300). happensAt(walking(mws1gt_6),300).
#show holdsAt/2.