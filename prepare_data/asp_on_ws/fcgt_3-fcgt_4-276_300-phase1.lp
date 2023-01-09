
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

	orientation(fcgt_3,164,276). coords(fcgt_3,98,182,276).  visible(fcgt_3,276). happensAt(walking(fcgt_3),276). orientation(fcgt_4,166,276). coords(fcgt_4,81,205,276).  visible(fcgt_4,276). happensAt(inactive(fcgt_4),276).
orientation(fcgt_3,164,277). coords(fcgt_3,98,182,277).  visible(fcgt_3,277). happensAt(walking(fcgt_3),277). orientation(fcgt_4,166,277). coords(fcgt_4,81,205,277).  visible(fcgt_4,277). happensAt(inactive(fcgt_4),277).
orientation(fcgt_3,164,278). coords(fcgt_3,98,182,278).  visible(fcgt_3,278). happensAt(walking(fcgt_3),278). orientation(fcgt_4,166,278). coords(fcgt_4,81,205,278).  visible(fcgt_4,278). happensAt(inactive(fcgt_4),278).
orientation(fcgt_3,164,279). coords(fcgt_3,98,183,279).  visible(fcgt_3,279). happensAt(walking(fcgt_3),279). orientation(fcgt_4,166,279). coords(fcgt_4,81,205,279).  visible(fcgt_4,279). happensAt(inactive(fcgt_4),279).
orientation(fcgt_3,164,280). coords(fcgt_3,98,183,280).  visible(fcgt_3,280). happensAt(walking(fcgt_3),280). orientation(fcgt_4,166,280). coords(fcgt_4,81,205,280).  visible(fcgt_4,280). happensAt(inactive(fcgt_4),280).
orientation(fcgt_3,164,281). coords(fcgt_3,98,184,281).  visible(fcgt_3,281). happensAt(walking(fcgt_3),281). orientation(fcgt_4,166,281). coords(fcgt_4,81,205,281).  visible(fcgt_4,281). happensAt(inactive(fcgt_4),281).
orientation(fcgt_3,164,282). coords(fcgt_3,98,185,282).  visible(fcgt_3,282). happensAt(walking(fcgt_3),282). orientation(fcgt_4,166,282). coords(fcgt_4,81,205,282).  visible(fcgt_4,282). happensAt(inactive(fcgt_4),282).
orientation(fcgt_3,164,283). coords(fcgt_3,98,185,283).  visible(fcgt_3,283). happensAt(walking(fcgt_3),283). orientation(fcgt_4,166,283). coords(fcgt_4,81,205,283).  visible(fcgt_4,283). happensAt(inactive(fcgt_4),283).
orientation(fcgt_3,164,284). coords(fcgt_3,98,185,284).  visible(fcgt_3,284). happensAt(walking(fcgt_3),284). orientation(fcgt_4,166,284). coords(fcgt_4,82,206,284).  visible(fcgt_4,284). happensAt(inactive(fcgt_4),284).
orientation(fcgt_3,164,285). coords(fcgt_3,98,185,285).  visible(fcgt_3,285). happensAt(walking(fcgt_3),285). orientation(fcgt_4,166,285). coords(fcgt_4,82,206,285).  visible(fcgt_4,285). happensAt(inactive(fcgt_4),285).
orientation(fcgt_3,164,286). coords(fcgt_3,98,185,286).  visible(fcgt_3,286). happensAt(walking(fcgt_3),286). orientation(fcgt_4,166,286). coords(fcgt_4,82,206,286).  visible(fcgt_4,286). happensAt(inactive(fcgt_4),286).
orientation(fcgt_3,164,287). coords(fcgt_3,98,185,287).  visible(fcgt_3,287). happensAt(walking(fcgt_3),287). orientation(fcgt_4,166,287). coords(fcgt_4,82,206,287).  visible(fcgt_4,287). happensAt(inactive(fcgt_4),287).
orientation(fcgt_3,164,288). coords(fcgt_3,98,185,288).  visible(fcgt_3,288). happensAt(inactive(fcgt_3),288). orientation(fcgt_4,166,288). coords(fcgt_4,82,206,288).  visible(fcgt_4,288). happensAt(inactive(fcgt_4),288).
orientation(fcgt_3,164,289). coords(fcgt_3,98,186,289).  visible(fcgt_3,289). happensAt(inactive(fcgt_3),289). orientation(fcgt_4,166,289). coords(fcgt_4,82,206,289).  visible(fcgt_4,289). happensAt(inactive(fcgt_4),289).
orientation(fcgt_3,164,290). coords(fcgt_3,98,186,290).  visible(fcgt_3,290). happensAt(inactive(fcgt_3),290). orientation(fcgt_4,166,290). coords(fcgt_4,82,206,290).  visible(fcgt_4,290). happensAt(inactive(fcgt_4),290).
orientation(fcgt_3,164,291). coords(fcgt_3,98,186,291).  visible(fcgt_3,291). happensAt(inactive(fcgt_3),291). orientation(fcgt_4,166,291). coords(fcgt_4,82,206,291).  visible(fcgt_4,291). happensAt(inactive(fcgt_4),291).
orientation(fcgt_3,164,292). coords(fcgt_3,98,186,292).  visible(fcgt_3,292). happensAt(inactive(fcgt_3),292). orientation(fcgt_4,166,292). coords(fcgt_4,82,206,292).  visible(fcgt_4,292). happensAt(inactive(fcgt_4),292).
orientation(fcgt_3,164,293). coords(fcgt_3,98,186,293).  visible(fcgt_3,293). happensAt(inactive(fcgt_3),293). orientation(fcgt_4,166,293). coords(fcgt_4,82,206,293).  visible(fcgt_4,293). happensAt(inactive(fcgt_4),293).
orientation(fcgt_3,164,294). coords(fcgt_3,98,186,294).  visible(fcgt_3,294). happensAt(inactive(fcgt_3),294). orientation(fcgt_4,166,294). coords(fcgt_4,82,206,294).  visible(fcgt_4,294). happensAt(inactive(fcgt_4),294).
orientation(fcgt_3,164,295). coords(fcgt_3,98,186,295).  visible(fcgt_3,295). happensAt(inactive(fcgt_3),295). orientation(fcgt_4,166,295). coords(fcgt_4,82,206,295).  visible(fcgt_4,295). happensAt(inactive(fcgt_4),295).
orientation(fcgt_3,164,296). coords(fcgt_3,98,186,296).  visible(fcgt_3,296). happensAt(inactive(fcgt_3),296). orientation(fcgt_4,166,296). coords(fcgt_4,82,206,296).  visible(fcgt_4,296). happensAt(inactive(fcgt_4),296).
orientation(fcgt_3,164,297). coords(fcgt_3,98,187,297).  visible(fcgt_3,297). happensAt(inactive(fcgt_3),297). orientation(fcgt_4,166,297). coords(fcgt_4,82,206,297).  visible(fcgt_4,297). happensAt(inactive(fcgt_4),297).
orientation(fcgt_3,164,298). coords(fcgt_3,98,187,298).  visible(fcgt_3,298). happensAt(inactive(fcgt_3),298). orientation(fcgt_4,166,298). coords(fcgt_4,82,206,298).  visible(fcgt_4,298). happensAt(inactive(fcgt_4),298).
orientation(fcgt_3,164,299). coords(fcgt_3,98,187,299).  visible(fcgt_3,299). happensAt(inactive(fcgt_3),299). orientation(fcgt_4,166,299). coords(fcgt_4,82,206,299).  visible(fcgt_4,299). happensAt(inactive(fcgt_4),299).
orientation(fcgt_3,164,300). coords(fcgt_3,98,187,300).  visible(fcgt_3,300). happensAt(inactive(fcgt_3),300). orientation(fcgt_4,166,300). coords(fcgt_4,82,206,300).  visible(fcgt_4,300). happensAt(inactive(fcgt_4),300).
#show holdsAt/2.