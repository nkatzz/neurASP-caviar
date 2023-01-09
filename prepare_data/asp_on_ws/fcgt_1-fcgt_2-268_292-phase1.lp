
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

	orientation(fcgt_1,151,268). coords(fcgt_1,135,137,268).  visible(fcgt_1,268). happensAt(walking(fcgt_1),268). orientation(fcgt_2,157,268). coords(fcgt_2,71,145,268).  visible(fcgt_2,268). happensAt(walking(fcgt_2),268).
orientation(fcgt_1,151,269). coords(fcgt_1,135,137,269).  visible(fcgt_1,269). happensAt(walking(fcgt_1),269). orientation(fcgt_2,157,269). coords(fcgt_2,71,145,269).  visible(fcgt_2,269). happensAt(walking(fcgt_2),269).
orientation(fcgt_1,151,270). coords(fcgt_1,133,137,270).  visible(fcgt_1,270). happensAt(walking(fcgt_1),270). orientation(fcgt_2,157,270). coords(fcgt_2,71,144,270).  visible(fcgt_2,270). happensAt(walking(fcgt_2),270).
orientation(fcgt_1,151,271). coords(fcgt_1,131,136,271).  visible(fcgt_1,271). happensAt(walking(fcgt_1),271). orientation(fcgt_2,157,271). coords(fcgt_2,73,144,271).  visible(fcgt_2,271). happensAt(walking(fcgt_2),271).
orientation(fcgt_1,151,272). coords(fcgt_1,130,137,272).  visible(fcgt_1,272). happensAt(walking(fcgt_1),272). orientation(fcgt_2,157,272). coords(fcgt_2,75,144,272).  visible(fcgt_2,272). happensAt(walking(fcgt_2),272).
orientation(fcgt_1,156,273). coords(fcgt_1,127,135,273).  visible(fcgt_1,273). happensAt(walking(fcgt_1),273). orientation(fcgt_2,157,273). coords(fcgt_2,75,143,273).  visible(fcgt_2,273). happensAt(walking(fcgt_2),273).
orientation(fcgt_1,156,274). coords(fcgt_1,125,133,274).  visible(fcgt_1,274). happensAt(walking(fcgt_1),274). orientation(fcgt_2,157,274). coords(fcgt_2,77,143,274).  visible(fcgt_2,274). happensAt(walking(fcgt_2),274).
orientation(fcgt_1,156,275). coords(fcgt_1,123,133,275).  visible(fcgt_1,275). happensAt(walking(fcgt_1),275). orientation(fcgt_2,157,275). coords(fcgt_2,77,143,275).  visible(fcgt_2,275). happensAt(walking(fcgt_2),275).
orientation(fcgt_1,153,276). coords(fcgt_1,122,133,276).  visible(fcgt_1,276). happensAt(walking(fcgt_1),276). orientation(fcgt_2,157,276). coords(fcgt_2,77,142,276).  visible(fcgt_2,276). happensAt(walking(fcgt_2),276).
orientation(fcgt_1,153,277). coords(fcgt_1,122,133,277).  visible(fcgt_1,277). happensAt(walking(fcgt_1),277). orientation(fcgt_2,157,277). coords(fcgt_2,77,141,277).  visible(fcgt_2,277). happensAt(walking(fcgt_2),277).
orientation(fcgt_1,153,278). coords(fcgt_1,121,133,278).  visible(fcgt_1,278). happensAt(walking(fcgt_1),278). orientation(fcgt_2,157,278). coords(fcgt_2,77,141,278).  visible(fcgt_2,278). happensAt(walking(fcgt_2),278).
orientation(fcgt_1,153,279). coords(fcgt_1,121,133,279).  visible(fcgt_1,279). happensAt(walking(fcgt_1),279). orientation(fcgt_2,157,279). coords(fcgt_2,78,140,279).  visible(fcgt_2,279). happensAt(walking(fcgt_2),279).
orientation(fcgt_1,153,280). coords(fcgt_1,120,131,280).  visible(fcgt_1,280). happensAt(walking(fcgt_1),280). orientation(fcgt_2,150,280). coords(fcgt_2,78,138,280).  visible(fcgt_2,280). happensAt(walking(fcgt_2),280).
orientation(fcgt_1,153,281). coords(fcgt_1,119,129,281).  visible(fcgt_1,281). happensAt(walking(fcgt_1),281). orientation(fcgt_2,150,281). coords(fcgt_2,78,138,281).  visible(fcgt_2,281). happensAt(walking(fcgt_2),281).
orientation(fcgt_1,153,282). coords(fcgt_1,118,127,282).  visible(fcgt_1,282). happensAt(walking(fcgt_1),282). orientation(fcgt_2,150,282). coords(fcgt_2,78,137,282).  visible(fcgt_2,282). happensAt(walking(fcgt_2),282).
orientation(fcgt_1,153,283). coords(fcgt_1,115,126,283).  visible(fcgt_1,283). happensAt(walking(fcgt_1),283). orientation(fcgt_2,150,283). coords(fcgt_2,81,136,283).  visible(fcgt_2,283). happensAt(walking(fcgt_2),283).
orientation(fcgt_1,153,284). coords(fcgt_1,113,126,284).  visible(fcgt_1,284). happensAt(walking(fcgt_1),284). orientation(fcgt_2,150,284). coords(fcgt_2,82,136,284).  visible(fcgt_2,284). happensAt(walking(fcgt_2),284).
orientation(fcgt_1,153,285). coords(fcgt_1,112,127,285).  visible(fcgt_1,285). happensAt(walking(fcgt_1),285). orientation(fcgt_2,150,285). coords(fcgt_2,82,136,285).  visible(fcgt_2,285). happensAt(walking(fcgt_2),285).
orientation(fcgt_1,153,286). coords(fcgt_1,111,127,286).  visible(fcgt_1,286). happensAt(walking(fcgt_1),286). orientation(fcgt_2,150,286). coords(fcgt_2,82,135,286).  visible(fcgt_2,286). happensAt(walking(fcgt_2),286).
orientation(fcgt_1,153,287). coords(fcgt_1,111,127,287).  visible(fcgt_1,287). happensAt(walking(fcgt_1),287). orientation(fcgt_2,150,287). coords(fcgt_2,83,135,287).  visible(fcgt_2,287). happensAt(walking(fcgt_2),287).
orientation(fcgt_1,153,288). coords(fcgt_1,110,127,288).  visible(fcgt_1,288). happensAt(walking(fcgt_1),288). orientation(fcgt_2,150,288). coords(fcgt_2,83,134,288).  visible(fcgt_2,288). happensAt(walking(fcgt_2),288).
orientation(fcgt_1,153,289). coords(fcgt_1,110,127,289).  visible(fcgt_1,289). happensAt(walking(fcgt_1),289). orientation(fcgt_2,150,289). coords(fcgt_2,83,134,289).  visible(fcgt_2,289). happensAt(walking(fcgt_2),289).
orientation(fcgt_1,153,290). coords(fcgt_1,110,128,290).  visible(fcgt_1,290). happensAt(walking(fcgt_1),290). orientation(fcgt_2,150,290). coords(fcgt_2,84,133,290).  visible(fcgt_2,290). happensAt(walking(fcgt_2),290).
orientation(fcgt_1,153,291). coords(fcgt_1,109,128,291).  visible(fcgt_1,291). happensAt(walking(fcgt_1),291). orientation(fcgt_2,150,291). coords(fcgt_2,84,133,291).  visible(fcgt_2,291). happensAt(walking(fcgt_2),291).
orientation(fcgt_1,153,292). coords(fcgt_1,107,128,292).  visible(fcgt_1,292). happensAt(walking(fcgt_1),292). orientation(fcgt_2,150,292). coords(fcgt_2,85,133,292).  visible(fcgt_2,292). happensAt(walking(fcgt_2),292).
#show holdsAt/2.