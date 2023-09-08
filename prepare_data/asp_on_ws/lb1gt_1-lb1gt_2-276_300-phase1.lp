
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

	orientation(lb1gt_1,151,276). coords(lb1gt_1,154,159,276).  visible(lb1gt_1,276). happensAt(active(lb1gt_1),276). orientation(lb1gt_2,146,276). coords(lb1gt_2,180,140,276).  visible(lb1gt_2,276). happensAt(walking(lb1gt_2),276).
orientation(lb1gt_1,151,277). coords(lb1gt_1,154,159,277).  visible(lb1gt_1,277). happensAt(active(lb1gt_1),277). orientation(lb1gt_2,146,277). coords(lb1gt_2,180,142,277).  visible(lb1gt_2,277). happensAt(walking(lb1gt_2),277).
orientation(lb1gt_1,151,278). coords(lb1gt_1,154,159,278).  visible(lb1gt_1,278). happensAt(active(lb1gt_1),278). orientation(lb1gt_2,146,278). coords(lb1gt_2,180,144,278).  visible(lb1gt_2,278). happensAt(walking(lb1gt_2),278).
orientation(lb1gt_1,151,279). coords(lb1gt_1,154,159,279).  visible(lb1gt_1,279). happensAt(active(lb1gt_1),279). orientation(lb1gt_2,146,279). coords(lb1gt_2,181,146,279).  visible(lb1gt_2,279). happensAt(walking(lb1gt_2),279).
orientation(lb1gt_1,151,280). coords(lb1gt_1,154,159,280).  visible(lb1gt_1,280). happensAt(active(lb1gt_1),280). orientation(lb1gt_2,146,280). coords(lb1gt_2,183,148,280).  visible(lb1gt_2,280). happensAt(walking(lb1gt_2),280).
orientation(lb1gt_1,151,281). coords(lb1gt_1,154,159,281).  visible(lb1gt_1,281). happensAt(active(lb1gt_1),281). orientation(lb1gt_2,146,281). coords(lb1gt_2,184,148,281).  visible(lb1gt_2,281). happensAt(walking(lb1gt_2),281).
orientation(lb1gt_1,151,282). coords(lb1gt_1,154,159,282).  visible(lb1gt_1,282). happensAt(active(lb1gt_1),282). orientation(lb1gt_2,146,282). coords(lb1gt_2,185,149,282).  visible(lb1gt_2,282). happensAt(walking(lb1gt_2),282).
orientation(lb1gt_1,151,283). coords(lb1gt_1,154,159,283).  visible(lb1gt_1,283). happensAt(active(lb1gt_1),283). orientation(lb1gt_2,146,283). coords(lb1gt_2,185,150,283).  visible(lb1gt_2,283). happensAt(walking(lb1gt_2),283).
orientation(lb1gt_1,151,284). coords(lb1gt_1,155,159,284).  visible(lb1gt_1,284). happensAt(active(lb1gt_1),284). orientation(lb1gt_2,146,284). coords(lb1gt_2,185,150,284).  visible(lb1gt_2,284). happensAt(walking(lb1gt_2),284).
orientation(lb1gt_1,151,285). coords(lb1gt_1,155,159,285).  visible(lb1gt_1,285). happensAt(active(lb1gt_1),285). orientation(lb1gt_2,146,285). coords(lb1gt_2,186,151,285).  visible(lb1gt_2,285). happensAt(walking(lb1gt_2),285).
orientation(lb1gt_1,151,286). coords(lb1gt_1,155,158,286).  visible(lb1gt_1,286). happensAt(active(lb1gt_1),286). orientation(lb1gt_2,146,286). coords(lb1gt_2,186,151,286).  visible(lb1gt_2,286). happensAt(walking(lb1gt_2),286).
orientation(lb1gt_1,151,287). coords(lb1gt_1,155,158,287).  visible(lb1gt_1,287). happensAt(active(lb1gt_1),287). orientation(lb1gt_2,146,287). coords(lb1gt_2,186,152,287).  visible(lb1gt_2,287). happensAt(walking(lb1gt_2),287).
orientation(lb1gt_1,151,288). coords(lb1gt_1,155,158,288).  visible(lb1gt_1,288). happensAt(active(lb1gt_1),288). orientation(lb1gt_2,146,288). coords(lb1gt_2,186,152,288).  visible(lb1gt_2,288). happensAt(walking(lb1gt_2),288).
orientation(lb1gt_1,151,289). coords(lb1gt_1,156,158,289).  visible(lb1gt_1,289). happensAt(active(lb1gt_1),289). orientation(lb1gt_2,146,289). coords(lb1gt_2,187,152,289).  visible(lb1gt_2,289). happensAt(walking(lb1gt_2),289).
orientation(lb1gt_1,151,290). coords(lb1gt_1,156,158,290).  visible(lb1gt_1,290). happensAt(walking(lb1gt_1),290). orientation(lb1gt_2,146,290). coords(lb1gt_2,189,152,290).  visible(lb1gt_2,290). happensAt(walking(lb1gt_2),290).
orientation(lb1gt_1,151,291). coords(lb1gt_1,156,158,291).  visible(lb1gt_1,291). happensAt(walking(lb1gt_1),291). orientation(lb1gt_2,146,291). coords(lb1gt_2,190,152,291).  visible(lb1gt_2,291). happensAt(walking(lb1gt_2),291).
orientation(lb1gt_1,151,292). coords(lb1gt_1,156,158,292).  visible(lb1gt_1,292). happensAt(walking(lb1gt_1),292). orientation(lb1gt_2,146,292). coords(lb1gt_2,191,153,292).  visible(lb1gt_2,292). happensAt(walking(lb1gt_2),292).
orientation(lb1gt_1,151,293). coords(lb1gt_1,156,159,293).  visible(lb1gt_1,293). happensAt(walking(lb1gt_1),293). orientation(lb1gt_2,146,293). coords(lb1gt_2,192,153,293).  visible(lb1gt_2,293). happensAt(walking(lb1gt_2),293).
orientation(lb1gt_1,151,294). coords(lb1gt_1,156,160,294).  visible(lb1gt_1,294). happensAt(walking(lb1gt_1),294). orientation(lb1gt_2,146,294). coords(lb1gt_2,192,153,294).  visible(lb1gt_2,294). happensAt(walking(lb1gt_2),294).
orientation(lb1gt_1,151,295). coords(lb1gt_1,156,160,295).  visible(lb1gt_1,295). happensAt(walking(lb1gt_1),295). orientation(lb1gt_2,146,295). coords(lb1gt_2,193,153,295).  visible(lb1gt_2,295). happensAt(active(lb1gt_2),295).
orientation(lb1gt_1,151,296). coords(lb1gt_1,157,160,296).  visible(lb1gt_1,296). happensAt(walking(lb1gt_1),296). orientation(lb1gt_2,146,296). coords(lb1gt_2,193,153,296).  visible(lb1gt_2,296). happensAt(active(lb1gt_2),296).
orientation(lb1gt_1,151,297). coords(lb1gt_1,158,161,297).  visible(lb1gt_1,297). happensAt(walking(lb1gt_1),297). orientation(lb1gt_2,146,297). coords(lb1gt_2,194,153,297).  visible(lb1gt_2,297). happensAt(active(lb1gt_2),297).
orientation(lb1gt_1,151,298). coords(lb1gt_1,159,162,298).  visible(lb1gt_1,298). happensAt(walking(lb1gt_1),298). orientation(lb1gt_2,146,298). coords(lb1gt_2,194,153,298).  visible(lb1gt_2,298). happensAt(active(lb1gt_2),298).
orientation(lb1gt_1,151,299). coords(lb1gt_1,159,163,299).  visible(lb1gt_1,299). happensAt(walking(lb1gt_1),299). orientation(lb1gt_2,146,299). coords(lb1gt_2,194,153,299).  visible(lb1gt_2,299). happensAt(active(lb1gt_2),299).
orientation(lb1gt_1,151,300). coords(lb1gt_1,160,164,300).  visible(lb1gt_1,300). happensAt(walking(lb1gt_1),300). orientation(lb1gt_2,146,300). coords(lb1gt_2,194,152,300).  visible(lb1gt_2,300). happensAt(active(lb1gt_2),300).
#show holdsAt/2.