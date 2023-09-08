
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

	orientation(rffgt_0,140,12). coords(rffgt_0,145,71,12).  visible(rffgt_0,12). happensAt(active(rffgt_0),12). orientation(rffgt_1,146,12). coords(rffgt_1,168,109,12).  visible(rffgt_1,12). happensAt(active(rffgt_1),12).
orientation(rffgt_0,140,13). coords(rffgt_0,145,71,13).  visible(rffgt_0,13). happensAt(active(rffgt_0),13). orientation(rffgt_1,146,13). coords(rffgt_1,167,109,13).  visible(rffgt_1,13). happensAt(active(rffgt_1),13).
orientation(rffgt_0,140,14). coords(rffgt_0,144,71,14).  visible(rffgt_0,14). happensAt(active(rffgt_0),14). orientation(rffgt_1,144,14). coords(rffgt_1,167,109,14).  visible(rffgt_1,14). happensAt(active(rffgt_1),14).
orientation(rffgt_0,140,15). coords(rffgt_0,144,70,15).  visible(rffgt_0,15). happensAt(active(rffgt_0),15). orientation(rffgt_1,144,15). coords(rffgt_1,167,109,15).  visible(rffgt_1,15). happensAt(active(rffgt_1),15).
orientation(rffgt_0,140,16). coords(rffgt_0,144,70,16).  visible(rffgt_0,16). happensAt(active(rffgt_0),16). orientation(rffgt_1,144,16). coords(rffgt_1,167,108,16).  visible(rffgt_1,16). happensAt(active(rffgt_1),16).
orientation(rffgt_0,140,17). coords(rffgt_0,144,70,17).  visible(rffgt_0,17). happensAt(active(rffgt_0),17). orientation(rffgt_1,144,17). coords(rffgt_1,167,108,17).  visible(rffgt_1,17). happensAt(active(rffgt_1),17).
orientation(rffgt_0,140,18). coords(rffgt_0,144,70,18).  visible(rffgt_0,18). happensAt(active(rffgt_0),18). orientation(rffgt_1,144,18). coords(rffgt_1,167,108,18).  visible(rffgt_1,18). happensAt(active(rffgt_1),18).
orientation(rffgt_0,140,19). coords(rffgt_0,144,70,19).  visible(rffgt_0,19). happensAt(active(rffgt_0),19). orientation(rffgt_1,141,19). coords(rffgt_1,167,107,19).  visible(rffgt_1,19). happensAt(active(rffgt_1),19).
orientation(rffgt_0,140,20). coords(rffgt_0,143,70,20).  visible(rffgt_0,20). happensAt(active(rffgt_0),20). orientation(rffgt_1,141,20). coords(rffgt_1,167,107,20).  visible(rffgt_1,20). happensAt(active(rffgt_1),20).
orientation(rffgt_0,140,21). coords(rffgt_0,143,70,21).  visible(rffgt_0,21). happensAt(active(rffgt_0),21). orientation(rffgt_1,141,21). coords(rffgt_1,166,106,21).  visible(rffgt_1,21). happensAt(active(rffgt_1),21).
orientation(rffgt_0,140,22). coords(rffgt_0,143,70,22).  visible(rffgt_0,22). happensAt(active(rffgt_0),22). orientation(rffgt_1,141,22). coords(rffgt_1,166,106,22).  visible(rffgt_1,22). happensAt(active(rffgt_1),22).
orientation(rffgt_0,140,23). coords(rffgt_0,143,70,23).  visible(rffgt_0,23). happensAt(active(rffgt_0),23). orientation(rffgt_1,141,23). coords(rffgt_1,166,106,23).  visible(rffgt_1,23). happensAt(active(rffgt_1),23).
orientation(rffgt_0,144,24). coords(rffgt_0,143,69,24).  visible(rffgt_0,24). happensAt(active(rffgt_0),24). orientation(rffgt_1,141,24). coords(rffgt_1,166,105,24).  visible(rffgt_1,24). happensAt(active(rffgt_1),24).
orientation(rffgt_0,144,25). coords(rffgt_0,143,69,25).  visible(rffgt_0,25). happensAt(active(rffgt_0),25). orientation(rffgt_1,138,25). coords(rffgt_1,166,105,25).  visible(rffgt_1,25). happensAt(active(rffgt_1),25).
orientation(rffgt_0,146,26). coords(rffgt_0,143,68,26).  visible(rffgt_0,26). happensAt(active(rffgt_0),26). orientation(rffgt_1,138,26). coords(rffgt_1,166,105,26).  visible(rffgt_1,26). happensAt(active(rffgt_1),26).
orientation(rffgt_0,146,27). coords(rffgt_0,143,68,27).  visible(rffgt_0,27). happensAt(active(rffgt_0),27). orientation(rffgt_1,138,27). coords(rffgt_1,165,105,27).  visible(rffgt_1,27). happensAt(active(rffgt_1),27).
orientation(rffgt_0,146,28). coords(rffgt_0,143,68,28).  visible(rffgt_0,28). happensAt(active(rffgt_0),28). orientation(rffgt_1,138,28). coords(rffgt_1,165,104,28).  visible(rffgt_1,28). happensAt(active(rffgt_1),28).
orientation(rffgt_0,146,29). coords(rffgt_0,143,68,29).  visible(rffgt_0,29). happensAt(active(rffgt_0),29). orientation(rffgt_1,138,29). coords(rffgt_1,165,104,29).  visible(rffgt_1,29). happensAt(active(rffgt_1),29).
orientation(rffgt_0,146,30). coords(rffgt_0,142,68,30).  visible(rffgt_0,30). happensAt(active(rffgt_0),30). orientation(rffgt_1,138,30). coords(rffgt_1,164,103,30).  visible(rffgt_1,30). happensAt(active(rffgt_1),30).
orientation(rffgt_0,146,31). coords(rffgt_0,142,68,31).  visible(rffgt_0,31). happensAt(active(rffgt_0),31). orientation(rffgt_1,138,31). coords(rffgt_1,164,104,31).  visible(rffgt_1,31). happensAt(active(rffgt_1),31).
orientation(rffgt_0,148,32). coords(rffgt_0,142,68,32).  visible(rffgt_0,32). happensAt(active(rffgt_0),32). orientation(rffgt_1,141,32). coords(rffgt_1,163,104,32).  visible(rffgt_1,32). happensAt(active(rffgt_1),32).
orientation(rffgt_0,148,33). coords(rffgt_0,141,68,33).  visible(rffgt_0,33). happensAt(active(rffgt_0),33). orientation(rffgt_1,146,33). coords(rffgt_1,163,104,33).  visible(rffgt_1,33). happensAt(active(rffgt_1),33).
orientation(rffgt_0,148,34). coords(rffgt_0,140,68,34).  visible(rffgt_0,34). happensAt(active(rffgt_0),34). orientation(rffgt_1,151,34). coords(rffgt_1,162,105,34).  visible(rffgt_1,34). happensAt(active(rffgt_1),34).
orientation(rffgt_0,148,35). coords(rffgt_0,139,67,35).  visible(rffgt_0,35). happensAt(active(rffgt_0),35). orientation(rffgt_1,151,35). coords(rffgt_1,162,105,35).  visible(rffgt_1,35). happensAt(active(rffgt_1),35).
orientation(rffgt_0,148,36). coords(rffgt_0,138,67,36).  visible(rffgt_0,36). happensAt(active(rffgt_0),36). orientation(rffgt_1,151,36). coords(rffgt_1,161,105,36).  visible(rffgt_1,36). happensAt(active(rffgt_1),36).
#show holdsAt/2.