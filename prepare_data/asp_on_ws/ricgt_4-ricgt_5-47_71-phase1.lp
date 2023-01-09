
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

	orientation(ricgt_4,172,47). coords(ricgt_4,30,125,47).  visible(ricgt_4,47). happensAt(walking(ricgt_4),47). orientation(ricgt_5,0,47). coords(ricgt_5,30,119,47).  visible(ricgt_5,47). happensAt(walking(ricgt_5),47).
orientation(ricgt_4,172,48). coords(ricgt_4,30,125,48).  visible(ricgt_4,48). happensAt(walking(ricgt_4),48). orientation(ricgt_5,0,48). coords(ricgt_5,30,119,48).  visible(ricgt_5,48). happensAt(walking(ricgt_5),48).
orientation(ricgt_4,172,49). coords(ricgt_4,30,125,49).  visible(ricgt_4,49). happensAt(walking(ricgt_4),49). orientation(ricgt_5,0,49). coords(ricgt_5,30,119,49).  visible(ricgt_5,49). happensAt(walking(ricgt_5),49).
orientation(ricgt_4,172,50). coords(ricgt_4,29,126,50).  visible(ricgt_4,50). happensAt(walking(ricgt_4),50). orientation(ricgt_5,0,50). coords(ricgt_5,30,119,50).  visible(ricgt_5,50). happensAt(walking(ricgt_5),50).
orientation(ricgt_4,172,51). coords(ricgt_4,29,127,51).  visible(ricgt_4,51). happensAt(walking(ricgt_4),51). orientation(ricgt_5,0,51). coords(ricgt_5,30,119,51).  visible(ricgt_5,51). happensAt(walking(ricgt_5),51).
orientation(ricgt_4,172,52). coords(ricgt_4,29,128,52).  visible(ricgt_4,52). happensAt(walking(ricgt_4),52). orientation(ricgt_5,0,52). coords(ricgt_5,30,119,52).  visible(ricgt_5,52). happensAt(walking(ricgt_5),52).
orientation(ricgt_4,172,53). coords(ricgt_4,29,128,53).  visible(ricgt_4,53). happensAt(walking(ricgt_4),53). orientation(ricgt_5,0,53). coords(ricgt_5,30,119,53).  visible(ricgt_5,53). happensAt(walking(ricgt_5),53).
orientation(ricgt_4,172,54). coords(ricgt_4,29,128,54).  visible(ricgt_4,54). happensAt(walking(ricgt_4),54). orientation(ricgt_5,0,54). coords(ricgt_5,29,120,54).  visible(ricgt_5,54). happensAt(walking(ricgt_5),54).
orientation(ricgt_4,172,55). coords(ricgt_4,29,128,55).  visible(ricgt_4,55). happensAt(walking(ricgt_4),55). orientation(ricgt_5,0,55). coords(ricgt_5,29,120,55).  visible(ricgt_5,55). happensAt(walking(ricgt_5),55).
orientation(ricgt_4,172,56). coords(ricgt_4,29,129,56).  visible(ricgt_4,56). happensAt(walking(ricgt_4),56). orientation(ricgt_5,0,56). coords(ricgt_5,29,121,56).  visible(ricgt_5,56). happensAt(walking(ricgt_5),56).
orientation(ricgt_4,172,57). coords(ricgt_4,29,129,57).  visible(ricgt_4,57). happensAt(walking(ricgt_4),57). orientation(ricgt_5,0,57). coords(ricgt_5,29,121,57).  visible(ricgt_5,57). happensAt(walking(ricgt_5),57).
orientation(ricgt_4,172,58). coords(ricgt_4,29,130,58).  visible(ricgt_4,58). happensAt(walking(ricgt_4),58). orientation(ricgt_5,0,58). coords(ricgt_5,29,121,58).  visible(ricgt_5,58). happensAt(walking(ricgt_5),58).
orientation(ricgt_4,172,59). coords(ricgt_4,28,130,59).  visible(ricgt_4,59). happensAt(walking(ricgt_4),59). orientation(ricgt_5,0,59). coords(ricgt_5,28,122,59).  visible(ricgt_5,59). happensAt(walking(ricgt_5),59).
orientation(ricgt_4,172,60). coords(ricgt_4,28,130,60).  visible(ricgt_4,60). happensAt(walking(ricgt_4),60). orientation(ricgt_5,0,60). coords(ricgt_5,28,123,60).  visible(ricgt_5,60). happensAt(walking(ricgt_5),60).
orientation(ricgt_4,172,61). coords(ricgt_4,28,130,61).  visible(ricgt_4,61). happensAt(walking(ricgt_4),61). orientation(ricgt_5,0,61). coords(ricgt_5,28,123,61).  visible(ricgt_5,61). happensAt(walking(ricgt_5),61).
orientation(ricgt_4,172,62). coords(ricgt_4,28,130,62).  visible(ricgt_4,62). happensAt(walking(ricgt_4),62). orientation(ricgt_5,0,62). coords(ricgt_5,28,123,62).  visible(ricgt_5,62). happensAt(walking(ricgt_5),62).
orientation(ricgt_4,172,63). coords(ricgt_4,28,130,63).  visible(ricgt_4,63). happensAt(walking(ricgt_4),63). orientation(ricgt_5,0,63). coords(ricgt_5,28,123,63).  visible(ricgt_5,63). happensAt(walking(ricgt_5),63).
orientation(ricgt_4,172,64). coords(ricgt_4,28,130,64).  visible(ricgt_4,64). happensAt(walking(ricgt_4),64). orientation(ricgt_5,0,64). coords(ricgt_5,28,123,64).  visible(ricgt_5,64). happensAt(walking(ricgt_5),64).
orientation(ricgt_4,172,65). coords(ricgt_4,28,130,65).  visible(ricgt_4,65). happensAt(walking(ricgt_4),65). orientation(ricgt_5,0,65). coords(ricgt_5,28,123,65).  visible(ricgt_5,65). happensAt(walking(ricgt_5),65).
orientation(ricgt_4,172,66). coords(ricgt_4,27,131,66).  visible(ricgt_4,66). happensAt(walking(ricgt_4),66). orientation(ricgt_5,0,66). coords(ricgt_5,28,123,66).  visible(ricgt_5,66). happensAt(walking(ricgt_5),66).
orientation(ricgt_4,172,67). coords(ricgt_4,27,131,67).  visible(ricgt_4,67). happensAt(walking(ricgt_4),67). orientation(ricgt_5,0,67). coords(ricgt_5,28,123,67).  visible(ricgt_5,67). happensAt(walking(ricgt_5),67).
orientation(ricgt_4,172,68). coords(ricgt_4,27,131,68).  visible(ricgt_4,68). happensAt(walking(ricgt_4),68). orientation(ricgt_5,0,68). coords(ricgt_5,28,123,68).  visible(ricgt_5,68). happensAt(walking(ricgt_5),68).
orientation(ricgt_4,172,69). coords(ricgt_4,27,131,69).  visible(ricgt_4,69). happensAt(walking(ricgt_4),69). orientation(ricgt_5,0,69). coords(ricgt_5,28,123,69).  visible(ricgt_5,69). happensAt(walking(ricgt_5),69).
orientation(ricgt_4,172,70). coords(ricgt_4,27,131,70).  visible(ricgt_4,70). happensAt(walking(ricgt_4),70). orientation(ricgt_5,0,70). coords(ricgt_5,28,123,70).  visible(ricgt_5,70). happensAt(walking(ricgt_5),70).
orientation(ricgt_4,172,71). coords(ricgt_4,27,131,71).  visible(ricgt_4,71). happensAt(walking(ricgt_4),71). orientation(ricgt_5,0,71). coords(ricgt_5,28,123,71).  visible(ricgt_5,71). happensAt(walking(ricgt_5),71).
#show holdsAt/2.