
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

	orientation(fcgt_1,151,268). coords(fcgt_1,135,137,268).  visible(fcgt_1,268). happensAt(walking(fcgt_1),268). orientation(fcgt_3,163,268). coords(fcgt_3,99,181,268).  visible(fcgt_3,268). happensAt(walking(fcgt_3),268).
orientation(fcgt_1,151,269). coords(fcgt_1,135,137,269).  visible(fcgt_1,269). happensAt(walking(fcgt_1),269). orientation(fcgt_3,166,269). coords(fcgt_3,99,181,269).  visible(fcgt_3,269). happensAt(walking(fcgt_3),269).
orientation(fcgt_1,151,270). coords(fcgt_1,133,137,270).  visible(fcgt_1,270). happensAt(walking(fcgt_1),270). orientation(fcgt_3,164,270). coords(fcgt_3,98,181,270).  visible(fcgt_3,270). happensAt(walking(fcgt_3),270).
orientation(fcgt_1,151,271). coords(fcgt_1,131,136,271).  visible(fcgt_1,271). happensAt(walking(fcgt_1),271). orientation(fcgt_3,164,271). coords(fcgt_3,98,181,271).  visible(fcgt_3,271). happensAt(walking(fcgt_3),271).
orientation(fcgt_1,151,272). coords(fcgt_1,130,137,272).  visible(fcgt_1,272). happensAt(walking(fcgt_1),272). orientation(fcgt_3,164,272). coords(fcgt_3,98,181,272).  visible(fcgt_3,272). happensAt(walking(fcgt_3),272).
orientation(fcgt_1,156,273). coords(fcgt_1,127,135,273).  visible(fcgt_1,273). happensAt(walking(fcgt_1),273). orientation(fcgt_3,164,273). coords(fcgt_3,98,181,273).  visible(fcgt_3,273). happensAt(walking(fcgt_3),273).
orientation(fcgt_1,156,274). coords(fcgt_1,125,133,274).  visible(fcgt_1,274). happensAt(walking(fcgt_1),274). orientation(fcgt_3,164,274). coords(fcgt_3,98,181,274).  visible(fcgt_3,274). happensAt(walking(fcgt_3),274).
orientation(fcgt_1,156,275). coords(fcgt_1,123,133,275).  visible(fcgt_1,275). happensAt(walking(fcgt_1),275). orientation(fcgt_3,164,275). coords(fcgt_3,98,181,275).  visible(fcgt_3,275). happensAt(walking(fcgt_3),275).
orientation(fcgt_1,153,276). coords(fcgt_1,122,133,276).  visible(fcgt_1,276). happensAt(walking(fcgt_1),276). orientation(fcgt_3,164,276). coords(fcgt_3,98,182,276).  visible(fcgt_3,276). happensAt(walking(fcgt_3),276).
orientation(fcgt_1,153,277). coords(fcgt_1,122,133,277).  visible(fcgt_1,277). happensAt(walking(fcgt_1),277). orientation(fcgt_3,164,277). coords(fcgt_3,98,182,277).  visible(fcgt_3,277). happensAt(walking(fcgt_3),277).
orientation(fcgt_1,153,278). coords(fcgt_1,121,133,278).  visible(fcgt_1,278). happensAt(walking(fcgt_1),278). orientation(fcgt_3,164,278). coords(fcgt_3,98,182,278).  visible(fcgt_3,278). happensAt(walking(fcgt_3),278).
orientation(fcgt_1,153,279). coords(fcgt_1,121,133,279).  visible(fcgt_1,279). happensAt(walking(fcgt_1),279). orientation(fcgt_3,164,279). coords(fcgt_3,98,183,279).  visible(fcgt_3,279). happensAt(walking(fcgt_3),279).
orientation(fcgt_1,153,280). coords(fcgt_1,120,131,280).  visible(fcgt_1,280). happensAt(walking(fcgt_1),280). orientation(fcgt_3,164,280). coords(fcgt_3,98,183,280).  visible(fcgt_3,280). happensAt(walking(fcgt_3),280).
orientation(fcgt_1,153,281). coords(fcgt_1,119,129,281).  visible(fcgt_1,281). happensAt(walking(fcgt_1),281). orientation(fcgt_3,164,281). coords(fcgt_3,98,184,281).  visible(fcgt_3,281). happensAt(walking(fcgt_3),281).
orientation(fcgt_1,153,282). coords(fcgt_1,118,127,282).  visible(fcgt_1,282). happensAt(walking(fcgt_1),282). orientation(fcgt_3,164,282). coords(fcgt_3,98,185,282).  visible(fcgt_3,282). happensAt(walking(fcgt_3),282).
orientation(fcgt_1,153,283). coords(fcgt_1,115,126,283).  visible(fcgt_1,283). happensAt(walking(fcgt_1),283). orientation(fcgt_3,164,283). coords(fcgt_3,98,185,283).  visible(fcgt_3,283). happensAt(walking(fcgt_3),283).
orientation(fcgt_1,153,284). coords(fcgt_1,113,126,284).  visible(fcgt_1,284). happensAt(walking(fcgt_1),284). orientation(fcgt_3,164,284). coords(fcgt_3,98,185,284).  visible(fcgt_3,284). happensAt(walking(fcgt_3),284).
orientation(fcgt_1,153,285). coords(fcgt_1,112,127,285).  visible(fcgt_1,285). happensAt(walking(fcgt_1),285). orientation(fcgt_3,164,285). coords(fcgt_3,98,185,285).  visible(fcgt_3,285). happensAt(walking(fcgt_3),285).
orientation(fcgt_1,153,286). coords(fcgt_1,111,127,286).  visible(fcgt_1,286). happensAt(walking(fcgt_1),286). orientation(fcgt_3,164,286). coords(fcgt_3,98,185,286).  visible(fcgt_3,286). happensAt(walking(fcgt_3),286).
orientation(fcgt_1,153,287). coords(fcgt_1,111,127,287).  visible(fcgt_1,287). happensAt(walking(fcgt_1),287). orientation(fcgt_3,164,287). coords(fcgt_3,98,185,287).  visible(fcgt_3,287). happensAt(walking(fcgt_3),287).
orientation(fcgt_1,153,288). coords(fcgt_1,110,127,288).  visible(fcgt_1,288). happensAt(walking(fcgt_1),288). orientation(fcgt_3,164,288). coords(fcgt_3,98,185,288).  visible(fcgt_3,288). happensAt(inactive(fcgt_3),288).
orientation(fcgt_1,153,289). coords(fcgt_1,110,127,289).  visible(fcgt_1,289). happensAt(walking(fcgt_1),289). orientation(fcgt_3,164,289). coords(fcgt_3,98,186,289).  visible(fcgt_3,289). happensAt(inactive(fcgt_3),289).
orientation(fcgt_1,153,290). coords(fcgt_1,110,128,290).  visible(fcgt_1,290). happensAt(walking(fcgt_1),290). orientation(fcgt_3,164,290). coords(fcgt_3,98,186,290).  visible(fcgt_3,290). happensAt(inactive(fcgt_3),290).
orientation(fcgt_1,153,291). coords(fcgt_1,109,128,291).  visible(fcgt_1,291). happensAt(walking(fcgt_1),291). orientation(fcgt_3,164,291). coords(fcgt_3,98,186,291).  visible(fcgt_3,291). happensAt(inactive(fcgt_3),291).
orientation(fcgt_1,153,292). coords(fcgt_1,107,128,292).  visible(fcgt_1,292). happensAt(walking(fcgt_1),292). orientation(fcgt_3,164,292). coords(fcgt_3,98,186,292).  visible(fcgt_3,292). happensAt(inactive(fcgt_3),292).
#show holdsAt/2.