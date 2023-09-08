
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

	orientation(br1gt_0,158,48). coords(br1gt_0,209,180,48).  visible(br1gt_0,48). happensAt(walking(br1gt_0),48). orientation(br1gt_1,164,48). coords(br1gt_1,125,193,48).  visible(br1gt_1,48). happensAt(walking(br1gt_1),48).
orientation(br1gt_0,154,49). coords(br1gt_0,208,178,49).  visible(br1gt_0,49). happensAt(walking(br1gt_0),49). orientation(br1gt_1,164,49). coords(br1gt_1,124,192,49).  visible(br1gt_1,49). happensAt(walking(br1gt_1),49).
orientation(br1gt_0,157,50). coords(br1gt_0,207,176,50).  visible(br1gt_0,50). happensAt(walking(br1gt_0),50). orientation(br1gt_1,164,50). coords(br1gt_1,124,191,50).  visible(br1gt_1,50). happensAt(walking(br1gt_1),50).
orientation(br1gt_0,157,51). coords(br1gt_0,205,174,51).  visible(br1gt_0,51). happensAt(walking(br1gt_0),51). orientation(br1gt_1,164,51). coords(br1gt_1,123,190,51).  visible(br1gt_1,51). happensAt(walking(br1gt_1),51).
orientation(br1gt_0,157,52). coords(br1gt_0,202,173,52).  visible(br1gt_0,52). happensAt(walking(br1gt_0),52). orientation(br1gt_1,164,52). coords(br1gt_1,123,189,52).  visible(br1gt_1,52). happensAt(walking(br1gt_1),52).
orientation(br1gt_0,157,53). coords(br1gt_0,198,171,53).  visible(br1gt_0,53). happensAt(walking(br1gt_0),53). orientation(br1gt_1,164,53). coords(br1gt_1,122,188,53).  visible(br1gt_1,53). happensAt(walking(br1gt_1),53).
orientation(br1gt_0,161,54). coords(br1gt_0,196,168,54).  visible(br1gt_0,54). happensAt(walking(br1gt_0),54). orientation(br1gt_1,164,54). coords(br1gt_1,121,188,54).  visible(br1gt_1,54). happensAt(walking(br1gt_1),54).
orientation(br1gt_0,161,55). coords(br1gt_0,192,165,55).  visible(br1gt_0,55). happensAt(walking(br1gt_0),55). orientation(br1gt_1,164,55). coords(br1gt_1,120,187,55).  visible(br1gt_1,55). happensAt(walking(br1gt_1),55).
orientation(br1gt_0,161,56). coords(br1gt_0,190,163,56).  visible(br1gt_0,56). happensAt(walking(br1gt_0),56). orientation(br1gt_1,164,56). coords(br1gt_1,118,187,56).  visible(br1gt_1,56). happensAt(walking(br1gt_1),56).
orientation(br1gt_0,161,57). coords(br1gt_0,187,161,57).  visible(br1gt_0,57). happensAt(walking(br1gt_0),57). orientation(br1gt_1,164,57). coords(br1gt_1,117,186,57).  visible(br1gt_1,57). happensAt(walking(br1gt_1),57).
orientation(br1gt_0,161,58). coords(br1gt_0,186,161,58).  visible(br1gt_0,58). happensAt(walking(br1gt_0),58). orientation(br1gt_1,164,58). coords(br1gt_1,116,186,58).  visible(br1gt_1,58). happensAt(walking(br1gt_1),58).
orientation(br1gt_0,161,59). coords(br1gt_0,185,160,59).  visible(br1gt_0,59). happensAt(walking(br1gt_0),59). orientation(br1gt_1,164,59). coords(br1gt_1,111,186,59).  visible(br1gt_1,59). happensAt(walking(br1gt_1),59).
orientation(br1gt_0,158,60). coords(br1gt_0,183,159,60).  visible(br1gt_0,60). happensAt(walking(br1gt_0),60). orientation(br1gt_1,164,60). coords(br1gt_1,109,185,60).  visible(br1gt_1,60). happensAt(walking(br1gt_1),60).
orientation(br1gt_0,158,61). coords(br1gt_0,183,159,61).  visible(br1gt_0,61). happensAt(walking(br1gt_0),61). orientation(br1gt_1,164,61). coords(br1gt_1,108,184,61).  visible(br1gt_1,61). happensAt(walking(br1gt_1),61).
orientation(br1gt_0,158,62). coords(br1gt_0,183,159,62).  visible(br1gt_0,62). happensAt(walking(br1gt_0),62). orientation(br1gt_1,164,62). coords(br1gt_1,108,184,62).  visible(br1gt_1,62). happensAt(walking(br1gt_1),62).
orientation(br1gt_0,158,63). coords(br1gt_0,184,158,63).  visible(br1gt_0,63). happensAt(walking(br1gt_0),63). orientation(br1gt_1,164,63). coords(br1gt_1,107,183,63).  visible(br1gt_1,63). happensAt(walking(br1gt_1),63).
orientation(br1gt_0,157,64). coords(br1gt_0,184,158,64).  visible(br1gt_0,64). happensAt(walking(br1gt_0),64). orientation(br1gt_1,164,64). coords(br1gt_1,107,183,64).  visible(br1gt_1,64). happensAt(walking(br1gt_1),64).
orientation(br1gt_0,157,65). coords(br1gt_0,184,157,65).  visible(br1gt_0,65). happensAt(walking(br1gt_0),65). orientation(br1gt_1,164,65). coords(br1gt_1,107,183,65).  visible(br1gt_1,65). happensAt(walking(br1gt_1),65).
orientation(br1gt_0,157,66). coords(br1gt_0,185,156,66).  visible(br1gt_0,66). happensAt(walking(br1gt_0),66). orientation(br1gt_1,164,66). coords(br1gt_1,106,183,66).  visible(br1gt_1,66). happensAt(walking(br1gt_1),66).
orientation(br1gt_0,157,67). coords(br1gt_0,186,154,67).  visible(br1gt_0,67). happensAt(walking(br1gt_0),67). orientation(br1gt_1,164,67). coords(br1gt_1,105,183,67).  visible(br1gt_1,67). happensAt(walking(br1gt_1),67).
orientation(br1gt_0,157,68). coords(br1gt_0,187,154,68).  visible(br1gt_0,68). happensAt(walking(br1gt_0),68). orientation(br1gt_1,164,68). coords(br1gt_1,104,183,68).  visible(br1gt_1,68). happensAt(walking(br1gt_1),68).
orientation(br1gt_0,157,69). coords(br1gt_0,187,152,69).  visible(br1gt_0,69). happensAt(walking(br1gt_0),69). orientation(br1gt_1,164,69). coords(br1gt_1,102,181,69).  visible(br1gt_1,69). happensAt(walking(br1gt_1),69).
orientation(br1gt_0,157,70). coords(br1gt_0,187,151,70).  visible(br1gt_0,70). happensAt(walking(br1gt_0),70). orientation(br1gt_1,164,70). coords(br1gt_1,100,180,70).  visible(br1gt_1,70). happensAt(walking(br1gt_1),70).
orientation(br1gt_0,157,71). coords(br1gt_0,188,151,71).  visible(br1gt_0,71). happensAt(walking(br1gt_0),71). orientation(br1gt_1,164,71). coords(br1gt_1,99,179,71).  visible(br1gt_1,71). happensAt(walking(br1gt_1),71).
orientation(br1gt_0,157,72). coords(br1gt_0,187,149,72).  visible(br1gt_0,72). happensAt(walking(br1gt_0),72). orientation(br1gt_1,164,72). coords(br1gt_1,97,178,72).  visible(br1gt_1,72). happensAt(walking(br1gt_1),72).
#show holdsAt/2.